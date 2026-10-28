"""
Assignment 2: training pipeline. Drone vs bird alerts at an airport perimeter.

Run with:  uv run python main.py train

Steps:
1. Download the Roboflow dataset.
2. Describe it (images, instances per class, object size).
3. Finetune yolo11n and copy the best weights to weights/best.pt.
4. Evaluate on the test split: finetuned model, COCO zero-shot, predict-nothing.
5. Cost sweep: pick one threshold, then a pass / review / fail band.
6. Save everything the dashboard needs to outputs/.
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "drone-vs-bird"
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "best.pt"

# Roboflow Universe dataset (public, not forked)
WORKSPACE, PROJECT, VERSION = "drone-vs-bird-4idpt", "drone-vs-bird-v3", 4
CLASSES = ["Bird", "Drone"]  # class 1 (Drone) is the POSITIVE class

# Training settings (CPU on a MacBook Air M2)
EPOCHS, IMGSZ, BATCH = 3, 416, 16

# Business assumptions, explained in REPORT.md.
# One "case" = one camera snapshot of a flying object near the runway.
COST_FN = 60_000    # drone missed -> spotted late by a pilot -> unplanned closure + diversions
COST_FP = 150       # bird sent as drone alert -> security patrol called out for nothing
COST_TP = 150       # real drone alert -> same patrol call-out (we pay it anyway)
COST_TN = 0         # bird ignored correctly
COST_REVIEW = 0.60  # operator looks at the clip: 1 minute at 36 EUR/hour
BASE_RATE = 0.002   # 1 in 500 flying-object alerts is a drone
CASES_PER_DAY = 1_500
REVIEW_SECONDS = 60
REVIEW_HOURS_AVAILABLE = 6  # spare operator time per day in the ops room
MAX_REVIEW_SHARE = REVIEW_HOURS_AVAILABLE * 3600 / REVIEW_SECONDS / CASES_PER_DAY

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# 1. Data --------------------------------------------------------------------

def download_dataset() -> Path:
    if not DATA_DIR.exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR))
    # Some labels are polygons, mixed with boxes in the same file. YOLO then
    # ignores the whole image, so convert every polygon to its bounding box.
    for f in DATA_DIR.glob("*/labels/*.txt"):
        lines = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) > 5:
                xs, ys = list(map(float, parts[1::2])), list(map(float, parts[2::2]))
                box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                parts = [parts[0]] + [f"{v:.6f}" for v in box]
            if parts:
                lines.append(" ".join(parts))
        f.write_text("\n".join(lines) + "\n")
    # Write our own data.yaml with absolute paths so it works from anywhere
    yaml_path = DATA_DIR / "data_abs.yaml"
    yaml_path.write_text(
        f"path: {DATA_DIR}\ntrain: train/images\nval: valid/images\n"
        f"test: test/images\nnames: {CLASSES}\n")
    return yaml_path


def describe_dataset() -> dict:
    """Images, instances per class and median object size, per split."""
    stats = {}
    for split in ["train", "valid", "test"]:
        labels = list((DATA_DIR / split / "labels").glob("*.txt"))
        instances = {c: 0 for c in CLASSES}
        images_with = {c: 0 for c in CLASSES}
        areas = {c: [] for c in CLASSES}
        for f in labels:
            seen = set()
            for line in f.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = CLASSES[int(parts[0])]
                xy = list(map(float, parts[1:]))
                if len(xy) == 4:  # box: x y w h
                    area = xy[2] * xy[3]
                else:             # polygon: x1 y1 x2 y2 ... -> bounding box
                    area = (max(xy[0::2]) - min(xy[0::2])) * (max(xy[1::2]) - min(xy[1::2]))
                instances[cls] += 1
                areas[cls].append(area)
                seen.add(cls)
            for cls in seen:
                images_with[cls] += 1
        stats[split] = {
            "images": len(labels),
            "instances": instances,
            "images_with": images_with,
            "median_box_area_pct": {c: round(float(np.median(a)) * 100, 2) for c, a in areas.items()},
            "share_boxes_under_1pct": {c: round(float(np.mean(np.array(a) < 0.01)), 2) for c, a in areas.items()},
        }
    print(json.dumps(stats, indent=2))
    return stats


# 2. Training ----------------------------------------------------------------

def finetune(data_yaml: Path) -> None:
    model = YOLO("yolo11n.pt")  # COCO-pretrained starting point
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                device="cpu", workers=2, seed=0, project=str(ROOT / "runs"),
                name="drone", exist_ok=True)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs" / "drone" / "weights" / "best.pt", WEIGHTS)


# 3. Evaluation --------------------------------------------------------------

def detection_metrics(data_yaml: Path) -> dict:
    """AP50, mAP50-95, per-class precision/recall and confusion matrix on test."""
    m = YOLO(str(WEIGHTS)).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                                device="cpu", project=str(ROOT / "runs"),
                                name="test", exist_ok=True)
    per_class = {}
    for i, cls in enumerate(CLASSES):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[cls] = {"precision": round(p, 3), "recall": round(r, 3),
                          "AP50": round(ap50, 3), "AP50-95": round(ap, 3)}
    # rows = predicted, cols = true; last row/col = background
    cm = m.confusion_matrix.matrix.astype(int).tolist()
    return {"mAP50": round(m.box.map50, 3), "mAP50-95": round(m.box.map, 3),
            "per_class": per_class,
            "confusion_matrix": {"labels": CLASSES + ["background"], "rows_pred_cols_true": cm}}


def image_scores(model_path: str, split: str, drone_ids: list[int]) -> pd.DataFrame:
    """One row per image: true label (1 = drone in image) and drone score.

    drone score = highest confidence of any box the model calls a drone
    (0 if it finds none). This is the number the thresholds act on.
    """
    model = YOLO(model_path)
    rows = []
    for img in sorted((DATA_DIR / split / "images").glob("*.jpg")):
        label_file = DATA_DIR / split / "labels" / (img.stem + ".txt")
        true_ids = {int(l.split()[0]) for l in label_file.read_text().splitlines() if l.strip()}
        r = model.predict(str(img), conf=0.001, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        confs = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) in drone_ids]
        rows.append({"image": img.name, "is_drone": int(1 in true_ids),
                     "drone_score": max(confs, default=0.0)})
    return pd.DataFrame(rows)


# 4. Costs -------------------------------------------------------------------

def band_rates(df: pd.DataFrame, low: float, high: float) -> dict:
    """Share of drones / birds that land in pass (< low), review, fail (>= high)."""
    out = {}
    for name, part in [("drone", df[df.is_drone == 1]), ("bird", df[df.is_drone == 0])]:
        s = part.drone_score
        out[name] = {"pass": float((s < low).mean()),
                     "review": float(((s >= low) & (s < high)).mean()),
                     "fail": float((s >= high).mean())}
    return out


def cost_per_1000(df: pd.DataFrame, low: float, high: float, p: float = BASE_RATE) -> dict:
    """Expected euros per 1,000 cases at the production base rate.

    Test-set rates are re-weighted to the real base rate, because the test set
    is 77% drones while real alerts are 0.2% drones.
    Review is assumed correct: a reviewed drone becomes a TP, a reviewed bird a TN.
    """
    r = band_rates(df, low, high)
    d, b = r["drone"], r["bird"]
    per_case = (p * (d["pass"] * COST_FN + d["fail"] * COST_TP + d["review"] * (COST_REVIEW + COST_TP))
                + (1 - p) * (b["pass"] * COST_TN + b["fail"] * COST_FP + b["review"] * COST_REVIEW))
    review_share = p * d["review"] + (1 - p) * b["review"]
    return {"cost_per_1000": round(per_case * 1000, 2), "review_share": round(review_share, 4),
            "drone_recall": round(d["fail"] + d["review"], 3),
            "bird_false_alarm": round(b["fail"], 4)}


def sweep_single(df: pd.DataFrame) -> pd.DataFrame:
    """Model decides alone: one threshold, no review band."""
    return pd.DataFrame([{"threshold": t, **cost_per_1000(df, t, t)} for t in THRESHOLDS])


def best_band(df: pd.DataFrame) -> tuple[float, float]:
    """Cheapest (low, high) pair whose review share fits the people available."""
    best = None
    for low in THRESHOLDS:
        for high in THRESHOLDS[THRESHOLDS >= low]:
            c = cost_per_1000(df, low, high)
            if c["review_share"] <= MAX_REVIEW_SHARE and (best is None or c["cost_per_1000"] < best[0]):
                best = (c["cost_per_1000"], float(low), float(high))
    return best[1], best[2]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    stats = describe_dataset()
    finetune(data_yaml)
    det = detection_metrics(data_yaml)

    # Image-level drone scores for the three models, on valid and test
    models = {
        "finetuned": (str(WEIGHTS), [1]),
        "coco_zero_shot": ("yolo11n.pt", [4, 33]),  # COCO has no drone: airplane + kite as proxy
    }
    scores = {}
    for name, (path, ids) in models.items():
        for split in ["valid", "test"]:
            scores[(name, split)] = image_scores(path, split, ids)
    scores[("finetuned", "valid")].to_csv(OUT / "scores_valid.csv", index=False)
    scores[("finetuned", "test")].to_csv(OUT / "scores_test.csv", index=False)

    # Thresholds are chosen on VALID, then reported on TEST
    valid, test = scores[("finetuned", "valid")], scores[("finetuned", "test")]
    curve_valid = sweep_single(valid)
    t_single = float(curve_valid.loc[curve_valid.cost_per_1000.idxmin(), "threshold"])
    low, high = best_band(valid)
    sweep_single(test).to_csv(OUT / "cost_curve_test.csv", index=False)
    curve_valid.to_csv(OUT / "cost_curve_valid.csv", index=False)

    # Baselines and alternatives on TEST, all at the same base rate
    coco_test = scores[("coco_zero_shot", "test")]
    coco_curve = sweep_single(scores[("coco_zero_shot", "valid")])
    t_coco = float(coco_curve.loc[coco_curve.cost_per_1000.idxmin(), "threshold"])
    comparison = {
        "do nothing / predict nothing": cost_per_1000(test, 1.01, 1.01),
        "person checks everything": cost_per_1000(test, 0.0, 1.01),
        f"COCO zero-shot alone (t={t_coco})": cost_per_1000(coco_test, t_coco, t_coco),
        f"finetuned alone (t={t_single})": cost_per_1000(test, t_single, t_single),
        f"finetuned + review band ({low}-{high})": cost_per_1000(test, low, high),
    }
    print(json.dumps(comparison, indent=2))

    review_share = comparison[f"finetuned + review band ({low}-{high})"]["review_share"]
    reviews_per_day = review_share * CASES_PER_DAY
    metrics = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "splits": stats},
        "training": {"model": "yolo11n", "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "device": "cpu"},
        "detection_test": det,
        "costs": {"FN": COST_FN, "FP": COST_FP, "TP": COST_TP, "TN": COST_TN, "review": COST_REVIEW,
                  "base_rate": BASE_RATE, "cases_per_day": CASES_PER_DAY,
                  "review_seconds": REVIEW_SECONDS, "max_review_share": round(MAX_REVIEW_SHARE, 3)},
        "chosen": {"single_threshold": t_single, "low": low, "high": high},
        "comparison_test": comparison,
        "workload": {"reviews_per_day": round(reviews_per_day, 1),
                     "review_hours_per_day": round(reviews_per_day * REVIEW_SECONDS / 3600, 2)},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Cost-against-threshold plot (model alone), chosen threshold marked
    curve_test = pd.read_csv(OUT / "cost_curve_test.csv")
    plt.plot(curve_test.threshold, curve_test.cost_per_1000)
    plt.axvline(t_single, color="red", linestyle="--", label=f"chosen t = {t_single}")
    plt.yscale("log")
    plt.xlabel("drone confidence threshold")
    plt.ylabel("expected cost per 1,000 alerts (EUR, log)")
    plt.legend()
    plt.savefig(OUT / "cost_curve.png", dpi=120)
    print("Done. Weights in weights/best.pt, results in outputs/.")


if __name__ == "__main__":
    main()
