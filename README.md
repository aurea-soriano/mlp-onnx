# MLP → ONNX → HTML (entrenamiento local)

Ejemplo didáctico: entrenar localmente una **MLP** con scikit-learn usando el dataset Iris, exportar el modelo a **ONNX** y hacer inferencia desde una página HTML.



## Flujo

```text
Tu computadora                              
┌──────────────────────────────────────┐   
│ train.py                             │
│   Iris → StandardScaler → MLP        │
│                  ↓                   │
│          docs/model.onnx             │
│                  ↓                   │
│ docs/index.html + ONNX Runtime Web   │
└──────────────────────────────────────┘
```

Arquitectura: `4 variables → 16 neuronas ReLU → 8 neuronas ReLU → 3 especies`. El `StandardScaler` se exporta **dentro del ONNX**: no es necesario calcular la normalización con JavaScript.

## Estructura del proyecto

```text
mlp-onnx-local/
├── train.py             # Entrena, exporta el pipeline y verifica ONNX
├── requirements.txt     # Dependencias del entrenamiento local
├── docs/
│   ├── index.html       # Frontend HTML/CSS/JavaScript
│   └── model.onnx       # SE GENERA ejecutando python train.py
├── .gitignore
└── README.md
```

> **Importante:** `model.onnx` no está incluido en el ZIP: se genera en tu computadora en el paso 2.

## 1. Crear entorno virtual

Se recomienda Python 3.11 o 3.12.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

> Si Windows impide activar el entorno, puedes usar `.venv\Scripts\python.exe` en lugar de `python` para los comandos siguientes.

## 2. Entrenar localmente y generar el modelo ONNX

```bash
python train.py
```

Este script:

1. Carga Iris (150 observaciones; cuatro atributos; tres especies).
2. Divide el conjunto de entrenamiento/prueba, conservando proporciones de clases.
3. Ajusta el pipeline `StandardScaler → MLPClassifier(16, 8)`.
4. Exporta `docs/model.onnx`.
5. Compara las predicciones y probabilidades entre scikit-learn y ONNX Runtime.

Revisa en la terminal la exactitud en test y la confirmación de equivalencia.

## 3. Abrir la web en tu computadora

Desde la **raíz del proyecto**:

```bash
python -m http.server 8000 --directory docs
```

Abre **http://localhost:8000**.

Prueba, por ejemplo, estas medidas en centímetros: `5.1`, `3.5`, `1.4`, `0.2`. Pulsa **Clasificar flor** y observa las probabilidades de Setosa, Versicolor y Virginica.

**No abras** el HTML con doble clic (`file://`); ejecuta el servidor HTTP para que el navegador pueda cargar `model.onnx`.

El servidor solo distribuye archivos estáticos: **no** calcula las predicciones. La inferencia se realiza en el navegador mediante ONNX Runtime Web (WebAssembly). Se requiere acceso a Internet para descargar ONNX Runtime Web desde el CDN.



## Aspectos conceptuales

- **Entrenamiento:** `train.py` ejecutado en tu computadora, usando Python y scikit-learn.
- **Formato de intercambio:** ONNX almacena el pipeline `StandardScaler + MLP` como grafo computacional.
- **Inferencia:** `index.html` pasa un tensor `float32` de forma `[1, 4]` a ONNX Runtime Web y muestra probabilidades.
- **Hosting:** GitHub Pages, si se habilita, solo publica el HTML, el ONNX y otros recursos estáticos; no ejecuta Python.

## Referencias

- [skl2onnx](https://onnx.ai/sklearn-onnx/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [GitHub Pages: publicar desde una rama](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
