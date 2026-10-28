"""
Assignment 2: Gradio dashboard. Drone vs bird alerts at an airport perimeter.

Run with:  uv run python main.py app

Never trains anything. It only reads weights/best.pt and the files that
train.py saved in outputs/.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from train import cost_per_1000

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

model = YOLO(str(ROOT / "weights" / "best.pt"))
metrics = json.loads((OUT / "metrics.json").read_text())
test = pd.read_csv(OUT / "scores_test.csv")
curve = pd.read_csv(OUT / "cost_curve_test.csv")
CHOSEN = metrics["chosen"]
COSTS = metrics["costs"]


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return "FAIL: drone alert, send security patrol"
    if score >= low:
        return "REVIEW: operator checks the clip"
    return "PASS: bird, no action"


def predict(image, low, high):
    if image is None:
        return None, ""
    r = model.predict(image, conf=0.001, device="cpu", verbose=False)[0]
    drone_confs = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) == 1]
    score = max(drone_confs, default=0.0)
    # Only draw boxes above the low threshold so the picture is readable
    shown = r[r.boxes.conf >= max(low, 0.05)]
    return shown.plot()[:, :, ::-1], f"Drone score {score:.3f} -> {decide(score, low, high)}"


def band_view(low, high):
    """Confusion matrix (test set), cost per 1,000 and review share for a band."""
    if low > high:
        low, high = high, low
    counts = []
    for name, part in [("Drone (positive)", test[test.is_drone == 1]), ("Bird", test[test.is_drone == 0])]:
        s = part.drone_score
        counts.append({"actual": name, "pass": int((s < low).sum()),
                       "review": int(((s >= low) & (s < high)).sum()), "fail": int((s >= high).sum())})
    c = cost_per_1000(test, low, high, COSTS["base_rate"])
    reviews_day = c["review_share"] * COSTS["cases_per_day"]
    summary = (f"**Expected cost per 1,000 alerts:** EUR {c['cost_per_1000']:,.0f}  \n"
               f"**Share sent to review:** {c['review_share']:.1%} "
               f"(= {reviews_day:.0f} clips/day, {reviews_day * COSTS['review_seconds'] / 3600:.1f} operator hours; "
               f"capacity {COSTS['max_review_share']:.0%})  \n"
               f"**Drones caught (fail + review):** {c['drone_recall']:.1%}  \n"
               f"**Birds auto-flagged as drones:** {c['bird_false_alarm']:.2%}")
    return pd.DataFrame(counts), summary, cost_plot(low, high)


def cost_plot(low, high):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(curve.threshold, curve.cost_per_1000, label="model alone (one threshold)")
    ax.axvline(CHOSEN["single_threshold"], color="red", linestyle="--",
               label=f"chosen threshold {CHOSEN['single_threshold']}")
    ax.axvspan(low, high, color="orange", alpha=0.2, label="current review band")
    ax.set_yscale("log")
    ax.set_xlabel("drone confidence threshold")
    ax.set_ylabel("EUR per 1,000 alerts (log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


per_class = pd.DataFrame(metrics["detection_test"]["per_class"]).T.reset_index(names="class")
comparison = pd.DataFrame(metrics["comparison_test"]).T.reset_index(names="policy")

with gr.Blocks(title="Drone vs bird alerts") as demo:
    gr.Markdown("# Drone vs bird: airport perimeter alerts\nPositive class = **Drone**. "
                "Each case is one camera snapshot of a flying object.")
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN["low"], step=0.01, label="Low threshold (below = pass)")
        high = gr.Slider(0, 1, value=CHOSEN["high"], step=0.01, label="High threshold (above = fail)")

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload image")
            img_out = gr.Image(label="Predictions")
        verdict = gr.Markdown()
        img_in.change(predict, [img_in, low, high], [img_out, verdict])

    with gr.Tab("Thresholds and cost"):
        cm = gr.Dataframe(label="Confusion matrix on the test set (images)")
        summary = gr.Markdown()
        plot = gr.Plot(label="Cost against threshold")
        for s in (low, high):
            s.change(band_view, [low, high], [cm, summary, plot])
        demo.load(band_view, [low, high], [cm, summary, plot])

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Test set detection: **mAP50 {metrics['detection_test']['mAP50']}**, "
                    f"**mAP50-95 {metrics['detection_test']['mAP50-95']}**")
        gr.Dataframe(per_class, label="Per class (test)")
        gr.Dataframe(comparison, label="Policies compared on the test set (production base rate)")

if __name__ == "__main__":
    demo.launch()
