# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Especificar la direccion de mi dataset
directorio_dataset = 'dataset'

# Configurar el generador de datos para el conjunto de entrenamiento y validación
# Usaremos un 80% para entrenamiento y un 20% para validación
datagen = ImageDataGenerator(
    rescale=1./255,         # Normalizar los valores de los píxeles a [0, 1]
    validation_split=0.2      # 20% de los datos se utilizarán para la validación
)

# Generador para el conjunto de entrenamiento
train_generator = datagen.flow_from_directory(
    directorio_dataset,
    target_size=(28, 28),      # Redimensionar las imágenes al tamaño esperado por tu modelo
    batch_size=10,              # Tamaño del lote de imágenes que se generarán
    class_mode='categorical',   #categorical para clasificación multiclase
    color_mode='grayscale',     # Imagenes en escala de grises
    subset='training'           # Especifica que este es el conjunto de entrenamiento
)

# Generador para el conjunto de validación
validation_generator = datagen.flow_from_directory(
    directorio_dataset,
    target_size=(28, 28),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',
    shuffle=False
)

# Obtener las etiquetas de clase del generador
clases = train_generator.class_indices
print("Etiquetas de clase:", clases)
class_labels = list(clases.keys())

# Crear el modelo CNN
print("Creando el modelo CNN...")
model = Sequential([
    Conv2D(28, (3,3), activation='relu', input_shape=(28,28,1)), # 28,28 es el tamaño de las imágenes y 1 es el número de canales (escala de grises)
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo usando los generadores
print("Entrenando el modelo...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 8. Graficar resultados
print("Graficando resultados...")
fig2, ax2 = plt.subplots(1, 2, figsize=(12,4))

ax2[0].plot(history.history['accuracy'], label='Entrenamiento')
ax2[0].plot(history.history['val_accuracy'], label='Validación')
ax2[0].set_title('Precisión (Accuracy)')
ax2[0].legend()

ax2[1].plot(history.history['loss'], label='Entrenamiento')
ax2[1].plot(history.history['val_loss'], label='Validación')
ax2[1].set_title('Pérdida (Loss)')
ax2[1].legend()

plt.show()

# Evaluar el modelo
# Evaluar el conjunto de validación
print("Evaluando el modelo en el conjunto de validación...")
loss, acc = model.evaluate(validation_generator, verbose=0)
print(f"\nPrecisión en Validación: {acc:.4f}")
print(f"Pérdida en Validación: {loss:.4f}")

# Calcular y visualizar la Matriz de Confusión
print("Calculando y visualizando la Matriz de Confusión...")
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# identificar ejemplos bien y mal clasificados
print("Identificando ejemplos bien y mal clasificados...")
num_ejemplos_a_mostrar = 5
bien_clasificados_indices = []
mal_clasificados_indices = []

x_val_batch, y_val_batch = next(validation_generator)
y_val_true = np.argmax(y_val_batch, axis=1)
y_val_pred = np.argmax(model.predict(x_val_batch), axis=1)

for i in range(len(y_val_true)):
    if y_val_true[i] == y_val_pred[i] and len(bien_clasificados_indices) < num_ejemplos_a_mostrar:
        bien_clasificados_indices.append(i)
    elif y_val_true[i] != y_val_pred[i] and len(mal_clasificados_indices) < num_ejemplos_a_mostrar:
        mal_clasificados_indices.append(i)

# Visualizar ejemplos bien clasificados
if bien_clasificados_indices:
    plt.figure(figsize=(12, 3))
    for i, index in enumerate(bien_clasificados_indices):
        plt.subplot(1, num_ejemplos_a_mostrar, i + 1)
        plt.imshow(x_val_batch[index].squeeze(), cmap='gray')
        plt.title(f"True: {y_val_true[index]}, Pred: {y_val_pred[index]}")
        plt.axis('off')
    plt.suptitle("Ejemplos Bien Clasificados")
    plt.tight_layout()
    plt.show()

# Visualizar ejemplos mal clasificados
if mal_clasificados_indices:
    plt.figure(figsize=(12, 3))
    for i, index in enumerate(mal_clasificados_indices):
        plt.subplot(1, num_ejemplos_a_mostrar, i + 1)
        plt.imshow(x_val_batch[index].squeeze(), cmap='gray')
        plt.title(f"True: {y_val_true[index]}, Pred: {y_val_pred[index]}")
        plt.axis('off')
    plt.suptitle("Ejemplos Mal Clasificados")
    plt.tight_layout()
    plt.show()

model.save('modelo_cnn_numeros_propios.keras') # Guardar el modelo entrenado