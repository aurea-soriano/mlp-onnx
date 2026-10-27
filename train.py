"""Entrena una MLP Iris, exporta el pipeline a ONNX y valida equivalencia."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'docs' / 'model.onnx'

def main():
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(16, 8), activation='relu',
            max_iter=1500, random_state=42, early_stopping=False
        ))
    ])
    pipeline.fit(X_train, y_train)
    expected = pipeline.predict(X_test)
    print(f'Exactitud sklearn: {accuracy_score(y_test, expected):.3f}')

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[('features', FloatTensorType([None, 4]))],
        options={id(pipeline.named_steps['mlp']): {'zipmap': False}},
        target_opset=15
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(onnx_model.SerializeToString())

    session = ort.InferenceSession(str(OUTPUT), providers=['CPUExecutionProvider'])
    results = session.run(None, {session.get_inputs()[0].name: X_test})
    labels = results[0].reshape(-1)
    probabilities = results[1]
    assert np.array_equal(expected, labels), 'Predicciones sklearn/ONNX distintas'
    np.testing.assert_allclose(pipeline.predict_proba(X_test), probabilities, atol=1e-4, rtol=1e-4)
    print(f'Exactitud ONNX:    {accuracy_score(y_test, labels):.3f}')
    print(f'Modelo: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)')
    print(f'Entrada ONNX: {session.get_inputs()[0].name}')
    print(f'Salidas ONNX: {[x.name for x in session.get_outputs()]}')
    print('Verificación sklearn vs ONNX: OK')

if __name__ == '__main__':
    main()
