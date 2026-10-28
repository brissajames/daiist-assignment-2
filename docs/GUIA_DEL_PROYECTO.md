# Guía del proyecto (para estudiar y defenderlo en clase)

Este documento está en español para ti. Todo lo que se entrega (código,
`REPORT.md`, dashboard) está en inglés.

---

## 1. El proyecto en 30 segundos

Un aeropuerto tiene cámaras en el perímetro. Cada vez que algo vuela cerca de
la pista, salta una alerta con una foto. Hoy un operador mira cada foto y
decide: **¿es un pájaro (ignorar) o un dron (mandar a la patrulla)?**

Hemos entrenado un modelo YOLO que mira la foto y da una **puntuación de dron**
(de 0 a 1). Con dos umbrales decidimos:

- puntuación **< 0,20** → **PASS**: es un pájaro, no se hace nada.
- entre **0,20 y 0,40** → **REVIEW**: el operador mira el clip.
- **≥ 0,40** → **FAIL**: alerta de dron, se manda la patrulla.

**Clase positiva = Drone.**

---

## 2. Las decisiones que he tomado y por qué

### 2.1 Por qué este problema
El assignment exige que un falso positivo y un falso negativo cuesten
**distinto**. Aquí la diferencia es enorme:

| Error | Qué pasa | Coste |
|---|---|---|
| **Falso negativo** (dron que pasa como pájaro) | Nadie reacciona hasta que un piloto lo ve → cierre de pista no planificado, vuelos desviados | **60.000 €** |
| **Falso positivo** (pájaro tomado por dron) | La patrulla sale para nada (2 personas, 45 min) | **150 €** |
| Verdadero positivo | La patrulla sale (se paga igual, pero es lo correcto) | 150 € |
| Verdadero negativo | Nada | 0 € |
| Revisión humana | 1 minuto de un operador a 36 €/hora | 0,60 € |

**Un dron no detectado cuesta 400 veces más que una falsa alarma.**

**De dónde salen los 60.000 €:** en febrero de 2020 Barajas cerró ~1 hora por
drones y se desviaron 26 vuelos. Es una estimación del coste *extra* de un
cierre tarde y descontrolado. Si te preguntan: "es una estimación razonable,
y el informe dice qué pasaría si fuera mucho menor".

### 2.2 La tasa base (base rate)
En la vida real casi todas las alertas son pájaros. Supongo **1 dron cada 500
alertas (0,2 %)** y **1.500 alertas al día**.

**Muy importante:** en el test el 77 % de las imágenes tienen un dron. Si
calculáramos el coste con esa proporción, sería irreal. Por eso calculo, para
cada clase por separado, qué porcentaje cae en pass/review/fail, y luego lo
**re-pondero** con el 0,2 % real. (Es la función `cost_per_1000` en `train.py`.)

### 2.3 El dataset
- Roboflow Universe: `drone-vs-bird-4idpt / drone-vs-bird-v3 / versión 4`.
  Público, no le hice fork.
- 2 clases: Bird y Drone. Train 1.896, valid 379, test 253 imágenes.
- Elegí la **versión 4** porque no tiene aumentación (la 2 y la 3 triplican el
  train con giros y rotaciones, que solo hace el entrenamiento más lento).

**Qué implica el dataset (esto lo preguntan):**
1. **Objetos diminutos**: la caja típica ocupa menos del 1 % de la imagen. Es
   lo más difícil para un modelo pequeño.
2. **Cada imagen tiene solo una clase** (o pájaros o un dron) y **no hay
   cielos vacíos**. El modelo nunca ha visto "nada".
3. **Solo 59 imágenes de pájaros en test**: cada imagen mueve la tasa de falsa
   alarma 1,7 puntos. Es el número menos fiable.
4. **Algunas etiquetas eran polígonos** mezclados con cajas. YOLO descarta esas
   imágenes, así que el código convierte los polígonos en cajas.

### 2.4 El modelo y el fine-tuning
- **`yolo11n`** (detección, versión "nano", la más pequeña). Detección porque
  el dataset viene con cajas. Nano porque entrené en el portátil sin GPU.
- **Fine-tuning** = partir de un modelo ya entrenado en COCO (80 objetos
  comunes) y seguir entrenándolo con nuestras imágenes de drones y pájaros.
  Así no empieza de cero: ya sabe detectar formas y bordes.
- **Ajustes**: 3 épocas, imágenes a 416 px, batch 16, CPU de un MacBook Air M2,
  unos 7 minutos.
- **Por qué solo 3 épocas**: para que fuera rápido. Es una limitación que está
  en el informe: la mAP50 en validación iba 0,25 → 0,51 → 0,61, o sea que
  **seguía mejorando**. Con más épocas en GPU sería mejor.

  *Glosario rápido:* una **época** = el modelo ve todas las imágenes de train
  una vez. **Batch** = cuántas imágenes procesa a la vez.

### 2.5 De cajas a una decisión
YOLO da cajas con una confianza. Para decidir sobre la **foto entera**, la
**puntuación de dron** = la confianza más alta de cualquier caja que el
modelo llame "Drone" (0 si no hay ninguna).

### 2.6 Los umbrales se eligen en VALIDACIÓN y se reportan en TEST
Si eligiera el umbral mirando el test y luego reportara el resultado en el
mismo test, estaría haciendo trampa (el número saldría demasiado bonito).
Por eso: **barrido en validación → umbral elegido → resultado en test**.

### 2.7 La capacidad humana
El operador tiene unas **6 horas libres al día** = 360 revisiones de 1 minuto
de 1.500 alertas = **como mucho un 24 %** puede ir a revisión. El código busca
la pareja de umbrales más barata que cumpla ese límite.

---

## 3. Las líneas de código más importantes

### `train.py`

**Supuestos de negocio (arriba del archivo):**
```python
COST_FN = 60_000    # drone missed
COST_FP = 150       # bird sent as drone alert
COST_REVIEW = 0.60  # operator looks at the clip
BASE_RATE = 0.002   # 1 in 500 alerts is a drone
MAX_REVIEW_SHARE = REVIEW_HOURS_AVAILABLE * 3600 / REVIEW_SECONDS / CASES_PER_DAY
```
Todos los números de negocio están juntos para poder cambiarlos fácilmente.

**El fine-tuning (lo que el README llama "three lines"):**
```python
model = YOLO("yolo11n.pt")  # COCO-pretrained starting point
model.train(data=..., epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device="cpu", ...)
shutil.copy(".../best.pt", WEIGHTS)
```
`best.pt` = los pesos de la época con mejor resultado en validación. Se copian a
`weights/best.pt`, que es lo que carga el dashboard.

**Puntuación de dron por imagen (`image_scores`):**
```python
r = model.predict(str(img), conf=0.001, ...)
confs = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) in drone_ids]
"drone_score": max(confs, default=0.0)
```
`conf=0.001` para quedarnos con todas las cajas, incluso las de muy baja
confianza; así luego podemos probar cualquier umbral.

**El coste esperado (`cost_per_1000`), la fórmula clave:**
```python
per_case = (p * (d["pass"] * COST_FN + d["fail"] * COST_TP + d["review"] * (COST_REVIEW + COST_TP))
            + (1 - p) * (b["pass"] * COST_TN + b["fail"] * COST_FP + b["review"] * COST_REVIEW))
```
En palabras: *de cada alerta, con probabilidad p es un dron* y pagamos según
dónde cae (pass = lo perdimos, 60.000 €; fail o review = lo pillamos, 150 €
+ revisión). *Con probabilidad 1−p es un pájaro* y pagamos 150 € si cae en
fail y 0,60 € si cae en review. Asumimos que el operador siempre acierta.

**Barrido de un umbral (`sweep_single`)**: prueba umbrales de 0 a 1 de 0,01 en
0,01 con el modelo decidiendo solo (low = high) y se queda con el más barato.

**Banda de revisión (`best_band`)**: prueba todas las parejas (low, high) y se
queda con la más barata **cuyo % de revisión sea ≤ 24 %**.

**Baselines:**
```python
"coco_zero_shot": ("yolo11n.pt", [4, 33]),  # COCO has no drone: airplane + kite
cost_per_1000(test, 1.01, 1.01)  # do nothing: threshold above 1, nothing is ever flagged
```
COCO no tiene clase "dron", así que uso avión y cometa como lo más parecido.

### `app.py` (el dashboard)
- Carga `weights/best.pt` y los archivos de `outputs/`. **Nunca entrena.**
- Pestaña **Try an image**: subes foto → cajas + "PASS / REVIEW / FAIL".
- Pestaña **Thresholds and cost**: mueves los dos sliders y se actualizan la
  matriz de confusión, el coste por 1.000 alertas y el % a revisión. Debajo, la
  curva coste-umbral con el umbral elegido en rojo.
- Pestaña **Per-class results**: precisión/recall por clase y la tabla que
  compara todas las políticas.

---

## 4. Resultados

### 4.1 Detección en test
| | mAP50 | mAP50-95 |
|---|---|---|
| **Fine-tuned yolo11n** | **0,587** | **0,233** |
| COCO zero-shot | no tiene clase dron | — |
| Predict nothing | 0 | 0 |

### 4.2 Por clase (test)
| Clase | Precisión | Recall | AP50 |
|---|---|---|---|
| Bird | 0,42 | 0,49 | 0,38 |
| **Drone** | **0,78** | **0,75** | **0,79** |

- **El número bueno lo lleva Drone.** La media (0,587) esconde que los pájaros
  van mucho peor (vienen en bandadas de cajas diminutas).
- **El fallo principal**: confundir dron con pájaro. 124 de 237 drones se
  etiquetan como "bird". Es justo el error caro.

### 4.3 Comparación de modelos (coste por 1.000 alertas, en test)
| Modelo | Drones detectados | Pájaros marcados como dron | Coste |
|---|---|---|---|
| No hacer nada | 0 % | 0 % | 120.000 € |
| COCO zero-shot (t=0,38) | 27,8 % | 13,6 % | 106.980 € |
| **Fine-tuned (t=0,23)** | **83,5 %** | 23,7 % | **55.566 €** |

**El fine-tuned gana a los dos baselines**: reduce a la mitad el coste de no
hacer nada.

### 4.4 Con la banda de revisión
| Política | Drones detectados | % a revisión | Coste por 1.000 |
|---|---|---|---|
| No hacer nada | 0 % | 0 % | 120.000 € |
| Modelo solo (0,23) | 83,5 % | 0 % | 55.566 € |
| **Modelo + revisión (0,20–0,40)** | 87,1 % | 25,5 % | **23.490 €** |
| Una persona lo revisa todo | 100 % | 100 % | 900 € |

- Revisión en test = **382 clips al día = 6,4 horas** de operador (un poco más
  de las 6 h previstas: en validación salía 24 %, en test 25,5 %).

### 4.5 La conclusión (la parte que más puntúa)
**Que una persona lo revise todo es lo más barato**, porque un dron perdido
cuesta tanto que el 13 % de drones que el modelo deja pasar domina el coste.
Revisarlo todo son 25 horas de operador al día (unas 19 horas más que ahora,
2–3 personas más en turnos), unos 900 € al día. Eso es poquísimo comparado con
un solo dron perdido.

**Mi recomendación:** el modelo **no es lo bastante bueno para decidir solo**.
Con el personal actual, la banda 0,20–0,40 es la mejor opción y es mucho
mejor que nada. Pero lo correcto es **aumentar la capacidad de revisión**: con
el umbral bajo en 0,03 el modelo pilló el 100 % de drones del test mandando
el 52,6 % a revisión (~13 horas/día, 8.227 € por 1.000). El valor real del
modelo es **ordenar la cola** (lo más sospechoso primero) y **mandar la patrulla
al instante** cuando la puntuación es muy alta.

> Esto NO es un fracaso: el README dice que un resultado así "is a finding, as
> long as you report it". Lo importante es que la decisión sale de los números.

---

## 5. Preguntas que te pueden caer (y cómo responder)

**¿Por qué el falso negativo es peor?** Porque un dron no detectado obliga a
cerrar la pista tarde y desviar vuelos (~60.000 €); una falsa alarma es solo
mover la patrulla (150 €). 400 veces más.

**¿Qué es la tasa base y por qué importa?** La proporción real de drones en las
alertas (0,2 %). El test tiene 77 % drones, así que hay que re-ponderar; si no,
los costes salen falsos.

**¿Por qué eliges el umbral en validación y no en test?** Para no hacer
trampa: si lo eliges y lo evalúas en el mismo conjunto, el resultado sale
optimista.

**¿Qué supuesto cambiaría más tu respuesta?** Tasa base × coste de un dron
perdido (0,2 % × 60.000 = 120 € por alerta). Si bajara de ~43 € por alerta, el
modelo solo sería peor que no hacer nada.

**¿Qué es mAP50?** La precisión media de detección contando como acierto una
caja que solapa al menos un 50 % con la real. mAP50-95 es lo mismo pero
exigiendo solapes de 50 % a 95 %, más estricto.

**¿Precision vs recall?** Precision: de lo que el modelo dice que es dron, cuánto
lo es de verdad. Recall: de los drones reales, cuántos encuentra. Aquí importa
más el **recall de drones**.

**¿Por qué tres salidas y no dos?** Porque los casos dudosos es más barato
mandarlos a una persona (0,60 €) que equivocarse (150 € o 60.000 €).

**¿Es realista la carga de revisión?** 382 clips/día = 6,4 h, más o menos
2 h por turno de 8 h para el operador actual. Justo, pero posible.

**¿Qué mejorarías?** Entrenar más épocas (50+) a 640 px en GPU; añadir
imágenes de cielo vacío, aviones e insectos; tener más pájaros en test;
validar el coste de 60.000 € con el aeropuerto.

**¿Por qué yolo11n y no yolo11s?** Es más rápido y entrena en un portátil sin
GPU. Con GPU probaría yolo11s, que probablemente detecte mejor los objetos
pequeños.

---

## 6. Cómo se ejecuta

```bash
uv sync
uv run python main.py train   # descarga datos, entrena, evalúa, guarda outputs/
uv run python main.py app     # abre el dashboard en http://127.0.0.1:7860
uv run python check.py        # la comprobación oficial del profe
```

Archivos importantes:
- `train.py`: todo el pipeline.
- `app.py`: el dashboard.
- `weights/best.pt`: el modelo entrenado.
- `outputs/`: métricas, puntuaciones por imagen y la curva de coste.
- `REPORT.md`: el informe en inglés.
