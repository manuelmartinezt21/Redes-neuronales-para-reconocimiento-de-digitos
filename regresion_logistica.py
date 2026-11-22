"""
Regresión Logística para clasificación binaria de dígitos MNIST
================================================================
Proyecto: Red neuronal para reconocimiento de dígitos escritos a mano
Curso: Computación 2026 — Udelar FCIEN
Autor: Manuel Martínez

Descripción:
    Implementa una regresión logística con Scikit-learn para clasificar
    dos dígitos del conjunto MNIST (por defecto, el 0 y el 1).
    Se utilizan los archivos binarios del dataset ya descargados en data/.

Estructura:
    1. Carga del dataset MNIST desde archivos binarios
    2. Filtrado de dos dígitos a clasificar
    3. Visualización de ejemplos
    4. Entrenamiento del modelo de regresión logística
    5. Evaluación y reporte de resultados
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os

# ─── Parámetros configurables ──────────────────────────────────────────────────
DIGITO_A = 0          # Primer dígito a clasificar
DIGITO_B = 1          # Segundo dígito a clasificar
MAX_ITER = 1000       # Iteraciones máximas del optimizador
DATA_DIR = "data"     # Carpeta con los archivos MNIST

# ─── 1. Carga del dataset desde archivos binarios IDX ─────────────────────────

def leer_imagenes(ruta):
    """
    Lee imágenes del formato IDX3-ubyte de MNIST.

    Formato del archivo:
        [offset] [tipo]        [valor]          [descripción]
        0000     entero 32-bit 0x00000803        número mágico
        0004     entero 32-bit 60000 ó 10000     cantidad de imágenes
        0008     entero 32-bit 28               filas
        0012     entero 32-bit 28               columnas
        0016     byte sin signo ??              pixel[0][0][0]
        ...

    Retorna un array NumPy de forma (N, 784) con valores en [0, 1].
    """
    with open(ruta, "rb") as f:
        magico, n, filas, cols = struct.unpack(">IIII", f.read(16))
        assert magico == 2051, f"Número mágico incorrecto en {ruta}"
        datos = np.frombuffer(f.read(), dtype=np.uint8)
    # Redimensionar y normalizar al rango [0, 1]
    return datos.reshape(n, filas * cols).astype(np.float32) / 255.0


def leer_etiquetas(ruta):
    """
    Lee etiquetas del formato IDX1-ubyte de MNIST.

    Formato del archivo:
        [offset] [tipo]        [valor]          [descripción]
        0000     entero 32-bit 0x00000801        número mágico
        0004     entero 32-bit 60000 ó 10000     cantidad de etiquetas
        0008     byte sin signo ??              label[0]
        ...

    Retorna un array NumPy de forma (N,) con valores enteros entre 0 y 9.
    """
    with open(ruta, "rb") as f:
        magico, n = struct.unpack(">II", f.read(8))
        assert magico == 2049, f"Número mágico incorrecto en {ruta}"
        datos = np.frombuffer(f.read(), dtype=np.uint8)
    return datos


print("=" * 60)
print("REGRESIÓN LOGÍSTICA — CLASIFICACIÓN BINARIA MNIST")
print(f"Clasificando dígito {DIGITO_A} vs dígito {DIGITO_B}")
print("=" * 60)

# Rutas a los archivos
ruta_train_img = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
ruta_train_lbl = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
ruta_test_img  = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
ruta_test_lbl  = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

print("\n[1] Cargando datos MNIST...")
X_train_full = leer_imagenes(ruta_train_img)
y_train_full = leer_etiquetas(ruta_train_lbl)
X_test_full  = leer_imagenes(ruta_test_img)
y_test_full  = leer_etiquetas(ruta_test_lbl)

print(f"    Conjunto de entrenamiento: {X_train_full.shape[0]} imágenes")
print(f"    Conjunto de prueba:        {X_test_full.shape[0]} imágenes")
print(f"    Dimensión de cada imagen:  {X_train_full.shape[1]} píxeles (28×28)")

# ─── 2. Filtrado de dos dígitos ───────────────────────────────────────────────

print(f"\n[2] Filtrando dígitos {DIGITO_A} y {DIGITO_B}...")

mask_train = (y_train_full == DIGITO_A) | (y_train_full == DIGITO_B)
mask_test  = (y_test_full  == DIGITO_A) | (y_test_full  == DIGITO_B)

X_train = X_train_full[mask_train]
y_train = y_train_full[mask_train]
X_test  = X_test_full[mask_test]
y_test  = y_test_full[mask_test]

print(f"    Muestras de entrenamiento filtradas: {len(X_train)}")
print(f"      → Dígito {DIGITO_A}: {(y_train == DIGITO_A).sum()}")
print(f"      → Dígito {DIGITO_B}: {(y_train == DIGITO_B).sum()}")
print(f"    Muestras de prueba filtradas:        {len(X_test)}")

# ─── 3. Visualización de ejemplos ─────────────────────────────────────────────

print("\n[3] Generando visualización de ejemplos...")

fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
fig.suptitle(
    f"Ejemplos del conjunto de entrenamiento — Dígitos {DIGITO_A} y {DIGITO_B}",
    fontsize=12,
)

for col in range(10):
    # Fila superior: ejemplos del DIGITO_A
    idx_a = np.where(y_train == DIGITO_A)[0][col]
    axes[0, col].imshow(X_train[idx_a].reshape(28, 28), cmap="gray")
    axes[0, col].axis("off")
    if col == 0:
        axes[0, col].set_title(f"Dígito {DIGITO_A}", loc="left", fontsize=9)

    # Fila inferior: ejemplos del DIGITO_B
    idx_b = np.where(y_train == DIGITO_B)[0][col]
    axes[1, col].imshow(X_train[idx_b].reshape(28, 28), cmap="gray")
    axes[1, col].axis("off")
    if col == 0:
        axes[1, col].set_title(f"Dígito {DIGITO_B}", loc="left", fontsize=9)

plt.tight_layout()
plt.savefig("ejemplos_digitos.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Guardado: ejemplos_digitos.png")

# ─── 4. Entrenamiento del modelo ──────────────────────────────────────────────

print("\n[4] Entrenando modelo de regresión logística...")
print(f"    Parámetros: solver='lbfgs', max_iter={MAX_ITER}")

modelo = LogisticRegression(max_iter=MAX_ITER, solver="lbfgs", random_state=42)
modelo.fit(X_train, y_train)

print("    Entrenamiento completado.")

# ─── 5. Evaluación ────────────────────────────────────────────────────────────

print("\n[5] Evaluando modelo sobre el conjunto de prueba...")

y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n    ► Precisión (accuracy): {accuracy * 100:.2f}%\n")
print("    Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred,
        target_names=[f"Dígito {DIGITO_A}", f"Dígito {DIGITO_B}"]
    )
)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[f"Dígito {DIGITO_A}", f"Dígito {DIGITO_B}"]
)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(
    f"Matriz de confusión — Regresión Logística\n"
    f"Dígito {DIGITO_A} vs Dígito {DIGITO_B} (accuracy: {accuracy*100:.2f}%)"
)
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Guardado: matriz_confusion.png")

# ─── 6. Visualización de los pesos aprendidos ─────────────────────────────────

print("\n[6] Visualizando pesos aprendidos (w₁, …, w₇₈₄)...")

pesos = modelo.coef_[0].reshape(28, 28)

fig3, ax3 = plt.subplots(figsize=(4, 4))
im = ax3.imshow(pesos, cmap="RdBu", interpolation="nearest")
ax3.set_title(
    f"Pesos del modelo — Regresión Logística\n"
    f"Rojo = favorece dígito {DIGITO_B} · Azul = favorece dígito {DIGITO_A}"
)
ax3.axis("off")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("pesos_modelo.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Guardado: pesos_modelo.png")

# ─── 7. Ejemplos de predicciones incorrectas ──────────────────────────────────

errores = np.where(y_pred != y_test)[0]
print(f"\n[7] Ejemplos mal clasificados: {len(errores)} de {len(y_test)}")

if len(errores) > 0:
    n_mostrar = min(10, len(errores))
    fig4, axes4 = plt.subplots(1, n_mostrar, figsize=(max(2 * n_mostrar, 3), 2.5), squeeze=False)
    fig4.suptitle("Predicciones incorrectas", fontsize=11)

    for i, idx in enumerate(errores[:n_mostrar]):
        ax = axes4[0, i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"Real: {y_test[idx]}\nPred: {y_pred[idx]}",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("errores_clasificacion.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("    Guardado: errores_clasificacion.png")

print("\n" + "=" * 60)
print("Proceso completado.")
print("=" * 60)
