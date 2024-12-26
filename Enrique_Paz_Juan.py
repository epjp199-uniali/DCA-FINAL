import os
# Suprimir todos los mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Desactivar las operaciones personalizadas de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from random import sample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Input
from PIL import Image
from tensorflow.keras.activations import sigmoid, relu, tanh, elu
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical

# Configuración inicial
logging.disable(logging.WARNING)


# =========================
# Funciones auxiliares
# =========================

def cargarCifar10():
    """
    Carga y preprocesa el conjunto de datos CIFAR-10.
    """
    print("\n[Cargar CIFAR-10] Cargando datos...")
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    print(f"[Cargar CIFAR-10] Datos cargados. Tamaño del conjunto de entrenamiento: {X_train.shape}")
    print(f"[Cargar CIFAR-10] Tamaño del conjunto de prueba: {X_test.shape}")

    # Preprocesamiento
    print("[Cargar CIFAR-10] Normalizando datos y convirtiendo etiquetas a one-hot...")
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    Y_train_cat = to_categorical(Y_train, num_classes=10)
    Y_test_cat = to_categorical(Y_test, num_classes=10)

    print("[Cargar CIFAR-10] Datos preprocesados correctamente.")
    return X_train, Y_train, X_test, Y_test, Y_train_cat, Y_test_cat

def show_image(imagen, titulo):
    """
    Muestra una imagen con un título dado.
    """
    print(f"[Mostrar Imagen] Mostrando imagen: {titulo}")
    plt.figure()
    plt.title(titulo)
    plt.imshow(imagen)
    plt.axis('off')
    plt.show()

def graficarEvolucion(history, title="Evolución del entrenamiento"):
    """
    Muestra las gráficas de pérdida y precisión del entrenamiento.
    """
    print(f"[Graficar] Generando gráfica de evolución: {title}")
    plt.figure(figsize=(12, 5))

    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Entrenamiento', color='blue')
    plt.plot(history['val_loss'], label='Validación', color='red')
    plt.title(f'{title} - Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Entrenamiento', color='blue')
    plt.plot(history['val_accuracy'], label='Validación', color='red')
    plt.title(f'{title} - Precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.show()

def matrizConfusion(Y_test, predicciones, labels):
    """
    Genera y muestra la matriz de confusión.
    """
    print("[Matriz de Confusión] Generando matriz de confusión...")
    cm = confusion_matrix(Y_test, predicciones)  # Y_test y predicciones son vectores de índices
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Matriz de Confusión")
    plt.show()




def PromedioMatrizConfusion(Y_test, all_predictions, labels):
    """
    Calcula y muestra una matriz de confusión promedio basada en múltiples predicciones.
    """
    print("[Promedio Matriz de Confusión] Calculando matriz de confusión promedio...")
    
    # Convertir Y_test de one-hot a índices
    Y_true = Y_test.argmax(axis=1)
    
    # Inicializar acumulador de la matriz de confusión
    num_clases = len(labels)
    matriz_acumulada = np.zeros((num_clases, num_clases), dtype=np.float32)
    
    # Iterar sobre todas las predicciones
    for predicciones in all_predictions:
        cm = confusion_matrix(Y_true, predicciones, labels=range(num_clases))
        matriz_acumulada += cm
    
    # Promediar las matrices acumuladas
    matriz_promedio = matriz_acumulada / len(all_predictions)
    
    print("Matriz de Confusión Promedio (valores promediados):")
    
    # Mostrar la matriz promedio en el gráfico
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz_promedio, 
        display_labels=labels
    )
    
    # Crear la figura con un tamaño específico
    plt.figure(figsize=(10, 8))
    
    # Plotear con formato personalizado
    disp.plot(
        cmap=plt.cm.Blues,
        xticks_rotation='vertical',
        values_format='.1f'  # Formato con 1 decimal
    )
    
    plt.title("Matriz de Confusión Promedio")
    plt.tight_layout()  # Ajustar el layout
    plt.show()




# =========================
# Tarea A: Crear y entrenar MLP
# =========================

def crearMLP():
    """
    Define y compila un modelo MLP básico.
    """
    print("[Crear MLP] Definiendo modelo MLP...")
    model = Sequential([
        Input(shape=(32, 32, 3)),        # Capa de entrada explícita
        Flatten(),                       # Aplanar las imágenes sin input_shape
        Dense(32, activation='sigmoid'), # Capa oculta con 32 neuronas
        Dense(10, activation='softmax')  # Capa de salida
    ])
    print("[Crear MLP] Compilando modelo...")
    model.compile(optimizer=Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    print("[Crear MLP] Modelo definido y compilado.")
    return model

def entrenarMLP(model, X_train, Y_train, epocas, batch_size):
    """
    Entrena el modelo MLP y retorna el historial.
    """
    print(f"[Entrenar MLP] Iniciando entrenamiento")
    start_time = time.time()
    history = model.fit(X_train, Y_train,
                        validation_split=0.1,
                        epochs=epocas,
                        batch_size=batch_size,
                        verbose=1)
    training_time = time.time() - start_time
    print(f"[Entrenar MLP] Entrenamiento completado en {training_time:.2f} segundos.")
    return history, training_time

def evaluarMLP(model, X_test, Y_test, labels):
    """
    Evalúa el modelo con el conjunto de prueba.
    """
    print("[Evaluar MLP] Evaluando modelo con conjunto de prueba...")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
    predicciones = model.predict(X_test).argmax(axis=1)  # Convertir predicciones a índices
    Y_test_indices = Y_test.argmax(axis=1)  # Convertir Y_test a índices si está en one-hot encoding
    print(f"[Evaluar MLP] Pérdida en prueba: {test_loss:.4f}, Precisión en prueba: {test_accuracy:.4f}")
    matrizConfusion(Y_test_indices, predicciones, labels)


# =========================
# Tareas B y C: Ajuste de hiperparámetros
# =========================

def entrenarConEpocas(X_train, Y_train, epocas_lista, repeticiones=5):
    """
    Entrena el modelo con diferentes números de épocas (con repeticiones).
    """
    print("[Entrenar con Épocas] Ajustando número de épocas...")
    resultados = []
    for epocas in epocas_lista:
        print(f"[Entrenar con Épocas] Entrenando con {epocas} épocas...")
        historiales = []
        for _ in range(repeticiones):
            model = crearMLP()
            history, _ = entrenarMLP(model, X_train, Y_train, epocas, batch_size=32)
            historiales.append(history)
        # Promediamos los resultados de las repeticiones
        resultados.append((epocas, historiales))
    print("[Entrenar con Épocas] Ajuste completado.")
    return resultados

def entrenarConBatchSize(X_train, Y_train, batch_sizes, repeticiones=5):
    """
    Entrena el modelo con diferentes tamaños de lote (con repeticiones) y captura los tiempos de entrenamiento.
    """
    print("[Entrenar con Tamaño de Lote] Ajustando batch_size...")
    resultados = []
    tiempos = []  # Lista para almacenar los tiempos de entrenamiento
    for batch_size in batch_sizes:
        print(f"[Entrenar con Tamaño de Lote] Entrenando con batch_size={batch_size}...")
        historiales = []
        tiempos_batch = []  # Lista para almacenar los tiempos de cada repetición
        for _ in range(repeticiones):
            model = crearMLP()
            history, training_time = entrenarMLP(model, X_train, Y_train, epocas=10, batch_size=batch_size)
            tiempos_batch.append(training_time)  # Almacenamos el tiempo de cada repetición
            historiales.append(history)
        # Promediamos los resultados de las repeticiones
        resultados.append((batch_size, historiales))
        tiempos.append(np.mean(tiempos_batch))  # Promediamos el tiempo para el batch_size actual

    print("[Entrenar con Tamaño de Lote] Ajuste completado.")
    return resultados, tiempos

def graficarComparacionBatchSize(batch_sizes, tiempos_batch, test_accuracies):
    # Crear la figura para la gráfica
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ancho de las barras
    bar_width = 0.4

    # Posiciones de las barras
    indices = np.arange(len(batch_sizes))

    # Graficar tiempo de entrenamiento
    ax.bar(indices - bar_width / 2, tiempos_batch, bar_width, color='tab:blue', alpha=0.8, label='Tiempo de Entrenamiento (s)')

    # Graficar precisión en prueba
    ax.bar(indices + bar_width / 2, test_accuracies, bar_width, color='tab:orange', alpha=0.8, label='Precisión en Prueba')

    # Añadir título y etiquetas
    ax.set_title("Comparación entre Tiempo de Entrenamiento y Precisión", fontsize=16)
    ax.set_xlabel("Tamaño de Lote (batch_size)", fontsize=14)
    ax.set_ylabel("Valores", fontsize=14)

    # Configurar etiquetas del eje X
    ax.set_xticks(indices)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=12)

    # Mostrar valores encima de las barras
    for i, (t, acc) in enumerate(zip(tiempos_batch, test_accuracies)):
        ax.text(indices[i] - bar_width / 2, t + 0.1, f"{t:.2f}", ha='center', fontsize=10, color='blue')
        ax.text(indices[i] + bar_width / 2, acc + 0.01, f"{acc:.2f}", ha='center', fontsize=10, color='orange')

    # Añadir leyenda
    ax.legend(fontsize=12)

    # Ajustar diseño
    plt.tight_layout()

    # Mostrar gráfica
    plt.show()

# =========================
# Tarea D: Ajustar funciones de activación
# =========================


def probarActivaciones(X_train, Y_train_cat, X_test, Y_test_cat, funciones_activacion, labels):
    """
    Entrena y evalúa varios MLP con diferentes funciones de activación y genera
    una gráfica comparando la precisión y el tiempo de entrenamiento para cada función.
    También genera matrices de confusión.
    Args:
        X_train: Imágenes de entrenamiento.
        Y_train_cat: Etiquetas de entrenamiento (one-hot).
        X_test: Imágenes de prueba.
        Y_test_cat: Etiquetas de prueba (one-hot).
        funciones_activacion: Lista de nombres de funciones de activación a probar.
        labels: Lista de etiquetas para la matriz de confusión.
    """
    # Mapear nombres de funciones a las funciones reales
    activations_map = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh,
        'elu': elu
    }
    
    if funciones_activacion is None or not funciones_activacion:
        print("[Probar Activaciones] Usando funciones de activación predeterminadas: sigmoid, relu, tanh, elu.")
        funciones_activacion = activations_map.keys()  # Usar las claves de activations

    resultados_activaciones = {}

    for activation_name in funciones_activacion:
        activation_func = activations_map.get(activation_name)
        if activation_func is None:
            print(f"[Probar Activaciones] Función desconocida: {activation_name}. Omitiendo...")
            continue

        print(f"\n[Probar Activaciones] Entrenando con activación {activation_name}...")

        # Crear un modelo con la función de activación actual
        model = Sequential([
            Input(shape=(32, 32, 3)),        # Capa de entrada explícita
            Flatten(),
            Dense(32, activation=activation_func), 
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        history, training_time = entrenarMLP(model, X_train, Y_train_cat, epocas=10, batch_size=32)

        # Evaluar el modelo y generar la matriz de confusión
        print(f"[Probar Activaciones] Evaluando modelo con activación {activation_name}...")
        test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
        predicciones = model.predict(X_test).argmax(axis=1)

        # Generar matriz de confusión
        Y_test_indices = Y_test_cat.argmax(axis=1)  # Convertir de one-hot a índices
        print(f"[Probar Activaciones] Matriz de Confusión para {activation_name}:")
        matrizConfusion(Y_test_indices, predicciones, labels)

        # Guardar resultados
        resultados_activaciones[activation_name] = {
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }

    return resultados_activaciones



def graficarResultadosActivaciones(resultados):
    """
    Genera una gráfica de barras comparando la precisión y el tiempo de entrenamiento
    para diferentes funciones de activación.
    """
    print("[Graficar] Generando gráfica de comparación de funciones de activación...")

    activations = list(resultados.keys())
    accuracies = [resultados[act]['test_accuracy'] for act in activations]
    times = [resultados[act]['training_time'] for act in activations]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    indices = np.arange(len(activations))

    # Gráfico de precisión
    ax1.bar(indices - bar_width / 2, accuracies, bar_width, color='tab:blue', label='Precisión en Prueba')

    ax2 = ax1.twinx()  # Crear un segundo eje Y para el tiempo
    ax2.bar(indices + bar_width / 2, times, bar_width, color='tab:orange', label='Tiempo de Entrenamiento')

    # Añadir título y etiquetas
    ax1.set_title("Comparación de Funciones de Activación", fontsize=16)
    ax1.set_xlabel("Función de Activación", fontsize=14)
    ax1.set_ylabel("Precisión en Prueba", fontsize=14)
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=14)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(activations, fontsize=12)

    # Añadir leyenda
    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

# =========================
# Tarea E: Ajustar el número de neuronas
# =========================

def probarNeurona(X_train, Y_train_cat, X_test, Y_test_cat, num_neuronas_lista, labels):
    """
    Entrena y evalúa varios MLP con diferente número de neuronas en su capa oculta.
    """
    resultados_neuronas = {}
    
    for num_neuronas in num_neuronas_lista:
        print(f"\n[Probar Neuronas] Entrenando modelo con {num_neuronas} neuronas en la capa oculta...")

        # Crear modelo con el número actual de neuronas
        model = Sequential([
            Input(shape=(32, 32, 3)),        # Capa de entrada explícita
            Flatten(),
            Dense(num_neuronas, activation='relu'),  # Ajustar número de neuronas
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        history, training_time = entrenarMLP(model, X_train, Y_train_cat, epocas=10, batch_size=32)

        # Evaluar y guardar resultados
        test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
        predicciones = model.predict(X_test).argmax(axis=1)

        # Generar matriz de confusión
        Y_test_indices = Y_test_cat.argmax(axis=1)  # Convertir de one-hot a índices
        print(f"[Probar Neuronas] Matriz de Confusión para {num_neuronas} neuronas:")
        matrizConfusion(Y_test_indices, predicciones, labels)

        resultados_neuronas[num_neuronas] = {
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }
    
    return resultados_neuronas

def graficarResultadosNeuronas(resultados):
    """
    Genera una gráfica comparando la precisión y el tiempo de entrenamiento
    para diferentes números de neuronas en la capa oculta.
    """
    print("[Graficar] Comparando neuronas en capa oculta...")

    neuronas = list(resultados.keys())
    accuracies = [resultados[neurona]['test_accuracy'] for neurona in neuronas]
    times = [resultados[neurona]['training_time'] for neurona in neuronas]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    indices = np.arange(len(neuronas))

    # Gráfico de precisión
    ax1.bar(indices - bar_width / 2, accuracies, bar_width, color='tab:blue', label='Precisión en Prueba')

    # Gráfico de tiempo
    ax2 = ax1.twinx()
    ax2.bar(indices + bar_width / 2, times, bar_width, color='tab:orange', label='Tiempo de Entrenamiento')

    # Añadir título y etiquetas
    ax1.set_title("Comparación del Número de Neuronas", fontsize=16)
    ax1.set_xlabel("Número de Neuronas", fontsize=14)
    ax1.set_ylabel("Precisión en Prueba", fontsize=14)
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=14)

    ax1.set_xticks(indices)
    ax1.set_xticklabels([str(n) for n in neuronas], fontsize=12)

    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

# =========================
# Tarea F: Optimizar un MLP de dos o más capas
# =========================

def probarCapasOcultas(X_train, Y_train_cat, X_test, Y_test_cat, configuraciones, labels):
    """
    Entrena y evalúa modelos con diferente número y tamaño de capas ocultas.
    También genera matrices de confusión para cada configuración.
    """
    resultados_capas = {}

    for config in configuraciones:
        print(f"\n[Probar Capas Ocultas] Entrenando modelo con configuración: {config}...")
        
        # Crear modelo con configuración actual
        model = Sequential([
            Input(shape=(32, 32, 3)),        # Capa de entrada explícita
            Flatten(),
        ])
        for num_neuronas in config:
            model.add(Dense(num_neuronas, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        history, training_time = entrenarMLP(model, X_train, Y_train_cat, epocas=10, batch_size=32)

        # Evaluar el modelo
        print(f"[Probar Capas Ocultas] Evaluando modelo con configuración: {config}...")
        test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
        predicciones = model.predict(X_test).argmax(axis=1)

        # Generar matriz de confusión
        try:
            Y_test_indices = Y_test_cat.argmax(axis=1)  # Convertir de one-hot a índices
        except AttributeError:
            print("[Probar Capas Ocultas] Y_test_cat ya está en formato de índices.")
            Y_test_indices = Y_test_cat  # Si ya está en formato correcto

        print(f"[Probar Capas Ocultas] Matriz de Confusión para configuración: {config}")
        matrizConfusion(Y_test_indices, predicciones, labels)

        # Guardar resultados
        resultados_capas[str(config)] = {
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }
    
    return resultados_capas


def graficarResultadosCapas(resultados):
    """
    Genera una gráfica comparando la precisión y el tiempo de entrenamiento
    para diferentes configuraciones de capas ocultas.
    """
    print("[Graficar] Comparando configuraciones de capas ocultas...")

    configuraciones = list(resultados.keys())
    accuracies = [resultados[config]['test_accuracy'] for config in configuraciones]
    times = [resultados[config]['training_time'] for config in configuraciones]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bar_width = 0.4
    indices = np.arange(len(configuraciones))

    ax1.bar(indices - bar_width / 2, accuracies, bar_width, color='tab:blue', label='Precisión en Prueba')

    ax2 = ax1.twinx()
    ax2.bar(indices + bar_width / 2, times, bar_width, color='tab:orange', label='Tiempo de Entrenamiento')

    ax1.set_title("Comparación de Configuraciones de Capas Ocultas", fontsize=16)
    ax1.set_xlabel("Configuración (Capas Ocultas)", fontsize=14)
    ax1.set_ylabel("Precisión en Prueba", fontsize=14)
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=14)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(configuraciones, rotation=45, fontsize=12)

    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

# =========================
# Tarea G: CNN sencilla con Keras
# =========================
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def crearCNNBasica():
    """
    Crea un modelo CNN básico con dos capas Conv2D y sin MaxPooling2D.
    """
    print("[Crear CNN] Definiendo modelo CNN básico...")
    model = Sequential([
        Input(shape=(32, 32, 3)),  # Capa de entrada explícita
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def crearCNNConMaxPooling():
    """
    Crea un modelo CNN con dos capas Conv2D y MaxPooling2D.
    """
    print("[Crear CNN con MaxPooling] Definiendo modelo CNN con MaxPooling...")
    model = Sequential([
        Input(shape=(32, 32, 3)),   # Capa de entrada explícita
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def evaluarCNNBasica(X_train, Y_train_cat, X_test, Y_test_cat, epocas, labels):
    """
    Entrena y evalúa una CNN básica.
    """
    print("[Evaluar CNN Básica] Iniciando entrenamiento...")
    model = crearCNNBasica()
    history = model.fit(X_train, Y_train_cat, epochs=epocas, validation_split=0.1, batch_size=32, verbose=1)
    graficarEvolucion(history.history, title="CNN Básica")

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
    print(f"Precisión final en test: {test_accuracy:.4f}")
    
    # Generar las predicciones y la matriz de confusión
    predicciones = model.predict(X_test).argmax(axis=1)
    matrizConfusion(Y_test_cat.argmax(axis=1), predicciones, labels)
    
    return model

def evaluarCNNConMaxPooling(X_train, Y_train_cat, X_test, Y_test_cat, epocas, labels):
    """
    Entrena y evalúa una CNN con capas MaxPooling2D.
    """
    print("[Evaluar CNN Con MaxPooling] Iniciando entrenamiento...")
    model = crearCNNConMaxPooling()
    history = model.fit(X_train, Y_train_cat, epochs=epocas, validation_split=0.1, batch_size=32, verbose=1)
    graficarEvolucion(history.history, title="CNN Con MaxPooling")

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
    print(f"Precisión final en test: {test_accuracy:.4f}")
    
    # Generar las predicciones y la matriz de confusión
    predicciones = model.predict(X_test).argmax(axis=1)
    matrizConfusion(Y_test_cat.argmax(axis=1), predicciones, labels)
    
    return model


def graficarComparativa(batch_sizes, tiempos_batch, test_accuracies):
    """
    Genera una gráfica comparativa entre tiempo de entrenamiento y precisión
    para cada tamaño de lote.
    """
    # Crear la figura para la gráfica
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ancho de las barras
    bar_width = 0.4

    # Posiciones de las barras
    indices = np.arange(len(batch_sizes))

    # Graficar tiempo de entrenamiento
    ax.bar(indices - bar_width / 2, tiempos_batch, bar_width, color='tab:blue', alpha=0.8, label='Tiempo de Entrenamiento (s)')

    # Graficar precisión en prueba
    ax.bar(indices + bar_width / 2, test_accuracies, bar_width, color='tab:orange', alpha=0.8, label='Precisión en Prueba')

    # Añadir título y etiquetas
    ax.set_title("Comparación entre Tiempo de Entrenamiento y Precisión", fontsize=16)
    ax.set_xlabel("Modelo", fontsize=14)
    ax.set_ylabel("Valores", fontsize=14)

    # Configurar etiquetas del eje X
    ax.set_xticks(indices)
    ax.set_xticklabels(batch_sizes, fontsize=12)

    # Mostrar valores encima de las barras
    for i, (t, acc) in enumerate(zip(tiempos_batch, test_accuracies)):
        ax.text(indices[i] - bar_width / 2, t + 0.1, f"{t:.2f}", ha='center', fontsize=10, color='blue')
        ax.text(indices[i] + bar_width / 2, acc + 0.01, f"{acc:.2f}", ha='center', fontsize=10, color='orange')

    # Añadir leyenda
    ax.legend(fontsize=12)

    # Ajustar diseño
    plt.tight_layout()

    # Mostrar gráfica
    plt.show()


def compararModelos(X_train, Y_train_cat, X_test, Y_test_cat, epocas=10, labels = range(10)):
    """
    Compara el modelo básico (sin MaxPooling2D) con el modelo que usa MaxPooling2D.
    """
    resultados = {}
    tiempos_batch = []  # Lista para almacenar los tiempos de entrenamiento
    test_accuracies = []  # Lista para almacenar las precisiones de prueba

    # Entrenando modelo básico (sin MaxPooling)
    print("\n[Comparar Modelos] Evaluando modelo básico (sin MaxPooling2D)...")
    
    start_time = time.time()  # Iniciar el temporizador
    model_basico = evaluarCNNBasica(X_train, Y_train_cat, X_test, Y_test_cat, epocas, labels)
    training_time_basico = time.time() - start_time  # Calcular el tiempo de entrenamiento
    
    # Guardar resultados del modelo básico
    test_loss_basico, test_accuracy_basico = model_basico.evaluate(X_test, Y_test_cat, verbose=0)
    resultados['Basico'] = {
        'accuracy': test_accuracy_basico,
        'time': training_time_basico,  # Guardar el tiempo calculado
    }

    # Añadir el tiempo de entrenamiento y la precisión al listado
    tiempos_batch.append(training_time_basico)
    test_accuracies.append(test_accuracy_basico)

    # Entrenando modelo con MaxPooling
    print("\n[Comparar Modelos] Evaluando modelo con MaxPooling2D...")
    
    start_time = time.time()  # Iniciar el temporizador
    model_maxpool = evaluarCNNConMaxPooling(X_train, Y_train_cat, X_test, Y_test_cat, epocas, labels)
    training_time_maxpool = time.time() - start_time  # Calcular el tiempo de entrenamiento
    
    # Guardar resultados del modelo con MaxPooling
    test_loss_maxpool, test_accuracy_maxpool = model_maxpool.evaluate(X_test, Y_test_cat, verbose=0)
    resultados['MaxPooling'] = {
        'accuracy': test_accuracy_maxpool,
        'time': training_time_maxpool,  # Guardar el tiempo calculado
    }

    # Añadir el tiempo de entrenamiento y la precisión al listado
    tiempos_batch.append(training_time_maxpool)
    test_accuracies.append(test_accuracy_maxpool)

    # Devolver los resultados y las métricas
    return list(resultados.keys()), tiempos_batch, test_accuracies



# =========================
# Tarea H: Ajustar kernel_size
# =========================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def probarTamanosFiltro(X_train, Y_train_cat, X_test, Y_test_cat, kernel_sizes):
    """
    Entrena y evalúa varios modelos CNN con diferentes tamaños de filtro (kernel_size),
    y devuelve las predicciones y etiquetas verdaderas.
    """
    resultados_filtros = {}
    all_predictions = []  # Lista para almacenar las predicciones de todos los modelos
    all_true_labels = []  # Lista para almacenar las etiquetas verdaderas

    for kernel_size in kernel_sizes:
        print(f"\n[Probar kernel_size] Entrenando modelo con kernel_size={kernel_size}...")

        # Crear el modelo
        model = Sequential([
            Input(shape=(32, 32, 3)),
            Conv2D(16, kernel_size, activation='relu'),
            Conv2D(32, kernel_size, activation='relu'),
            Flatten(),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        history, training_time = entrenarMLP(model, X_train, Y_train_cat, epocas=10, batch_size=32)

        # Evaluar el modelo
        test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)

        # Realizar predicciones
        Y_pred = model.predict(X_test)
        Y_pred_classes = np.argmax(Y_pred, axis=1)  # Convertir las probabilidades en clases predichas
        Y_true = np.argmax(Y_test_cat, axis=1)  # Convertir las etiquetas verdaderas a clases

        # Guardar las predicciones y las etiquetas verdaderas
        all_predictions.append(Y_pred_classes)
        all_true_labels.append(Y_true)

        # Guardar los resultados del modelo
        resultados_filtros[str(kernel_size)] = {
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }

    return resultados_filtros, all_true_labels, all_predictions

def graficarResultadosKernel(resultados):
    """
    Genera una gráfica de barras comparando la precisión y el tiempo de entrenamiento
    para diferentes funciones de activación.
    """
    print("[Graficar] Generando gráfica de comparación de tamaños de kernel...")

    activations = list(resultados.keys())
    accuracies = [resultados[act]['test_accuracy'] for act in activations]
    times = [resultados[act]['training_time'] for act in activations]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    indices = np.arange(len(activations))

    # Gráfico de precisión
    ax1.bar(indices - bar_width / 2, accuracies, bar_width, color='tab:blue', label='Precisión en Prueba')

    ax2 = ax1.twinx()  # Crear un segundo eje Y para el tiempo
    ax2.bar(indices + bar_width / 2, times, bar_width, color='tab:orange', label='Tiempo de Entrenamiento')

    # Añadir título y etiquetas
    ax1.set_title("Comparación de tamaños de Kernel", fontsize=16)
    ax1.set_xlabel("Tamaños Kernel", fontsize=14)
    ax1.set_ylabel("Precisión en Prueba", fontsize=14)
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=14)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(activations, fontsize=12)

    # Añadir leyenda
    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()


# =========================
# Tarea I: Optimizar arquitectura
# =========================
def optimizarCNN(X_train, Y_train_cat, X_test, Y_test_cat, configuraciones, labels):
    """
    Prueba diferentes configuraciones de capas en la CNN para optimizar su rendimiento.
    """
    resultados_arquitectura = {}

    for config in configuraciones:
        print(f"\n[Optimizar CNN] Entrenando modelo con configuración: {config}...")
        
        # Crear modelo con la configuración actual
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))
        model.add(Conv2D(config[0], (3, 3), activation='relu'))
        if len(config) > 1:
            for num_filtros in config[1:]:
                model.add(Conv2D(num_filtros, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Entrenar el modelo
        history, training_time = entrenarMLP(model, X_train, Y_train_cat, epocas=10, batch_size=32)

        # Evaluar y guardar resultados
        test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
        resultados_arquitectura[str(config)] = {
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }

        # Generar las predicciones para la matriz de confusión
        predicciones = model.predict(X_test).argmax(axis=1)
        etiquetas_verdaderas = np.argmax(Y_test_cat, axis=1)  # Convertir etiquetas en formato de índice
        
        # Llamar a la función para generar la matriz de confusión
        matrizConfusion(etiquetas_verdaderas, predicciones, labels)

    return resultados_arquitectura

def graficarResultadosOpt(resultados):
    """
    Genera una gráfica de barras comparando la precisión y el tiempo de entrenamiento
    para diferentes funciones de activación.
    """
    print("[Graficar] Generando gráfica de comparación de configuraciones de optimizacion...")

    activations = list(resultados.keys())
    accuracies = [resultados[act]['test_accuracy'] for act in activations]
    times = [resultados[act]['training_time'] for act in activations]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    indices = np.arange(len(activations))

    # Gráfico de precisión
    ax1.bar(indices - bar_width / 2, accuracies, bar_width, color='tab:blue', label='Precisión en Prueba')

    ax2 = ax1.twinx()  # Crear un segundo eje Y para el tiempo
    ax2.bar(indices + bar_width / 2, times, bar_width, color='tab:orange', label='Tiempo de Entrenamiento')

    # Añadir título y etiquetas
    ax1.set_title("Comparación de configuraciones de optimizacion", fontsize=16)
    ax1.set_xlabel("Configuraciones", fontsize=14)
    ax1.set_ylabel("Precisión en Prueba", fontsize=14)
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=14)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(activations, fontsize=12)

    # Añadir leyenda
    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

# =========================
# Tarea J: Crear conjunto de prueba y evaluar generalización
# =========================

def cargarImagenesPropias(num_imagenes=15):
    """
    Carga un subconjunto de imágenes del dataset CIFAR-10 y las devuelve junto con sus etiquetas.

    Args:
        num_imagenes (int): Número de imágenes a cargar por categoría.

    Returns:
        X_custom (numpy array): Imágenes cargadas.
        Y_custom (numpy array): Etiquetas correspondientes a las imágenes.
    """
    # Categorías del dataset CIFAR-10
    categorias = ["airplane", "automobile", "bird", "cat", "deer", 
                  "dog", "frog", "horse", "ship", "truck"]
    
    # Cargar datos de CIFAR-10
    (X_train, Y_train), _ = cifar10.load_data()

    # Inicializar listas para almacenar las imágenes y etiquetas
    X_custom = []
    Y_custom = []

    # Guardar imágenes en las listas correspondientes
    for i, categoria in enumerate(categorias):
        # Filtrar imágenes de la clase actual
        indices_clase = (Y_train.flatten() == i)
        imagenes_clase = X_train[indices_clase][:num_imagenes]
        
        # Añadir las imágenes y sus etiquetas a las listas
        X_custom.extend(imagenes_clase)
        Y_custom.extend([i] * len(imagenes_clase))  # Etiquetas correspondientes a las imágenes

    # Convertir las listas a arrays de numpy
    X_custom = np.array(X_custom)
    Y_custom = np.array(Y_custom)

    # Normalizar las imágenes a rango [0, 1]
    X_custom = X_custom.astype("float32") / 255.0
    
    return X_custom, Y_custom


def evaluarGeneralizacion(model, X_custom, Y_custom, labels):
    """
    Evalúa la generalización del modelo usando un conjunto de imágenes propias.
    Args:
        model (keras.Model): Modelo entrenado.
        X_custom (np.array): Imágenes de prueba.
        Y_custom (np.array): Etiquetas de prueba.
        categorias (list): Lista de nombres de categorías.
    """
    print("[Evaluar Generalización] Evaluando con imágenes propias...")
    predicciones = model.predict(X_custom).argmax(axis=1)
    
    # Usar directamente Y_custom sin convertir a one-hot
    matrizConfusion(Y_custom, predicciones, labels)
    
    accuracy = np.mean(predicciones == Y_custom)
    print(f"Precisión en el conjunto propio: {accuracy * 100:.2f}%")


# =========================
# Tarea K: Resultados y experimentación
# =========================

def experimentarOptimizadores(X_train, Y_train, X_custom, Y_custom, categorias, optimizadores, labels):
    """
    Experimenta con diferentes optimizadores y evalúa el rendimiento en el conjunto de prueba personalizado.
    """
    resultados = {}

    for opt_name, optimizer in optimizadores.items():
        print(f"\n[Experimento] Optimizador: {opt_name}")

        model = Sequential([
            Input(shape=(32, 32, 3)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        training_time = time.time() - start_time

        predicciones = model.predict(X_custom).argmax(axis=1)
        accuracy = np.mean(predicciones == Y_custom.argmax(axis=1)) * 100  # Conversión a índices

        resultados[opt_name] = {
            'accuracy': accuracy,
            'time': training_time
        }

        matrizConfusion(Y_custom.argmax(axis=1), predicciones, labels)  # Conversión para la matriz

    return resultados


def graficarComparacionResultados(resultados, parametro):
    """
    Genera una gráfica comparativa de precisión y tiempo para diferentes configuraciones.
    Args:
        resultados (dict): Diccionario con configuraciones como claves y diccionarios con 'accuracy' y 'time' como valores.
        parametro (str): Nombre del parámetro que se está comparando (para el título y el eje X).
    """
    print(f"[Graficar Comparación] Generando gráfica para el parámetro: {parametro}...")
    configuraciones = list(resultados.keys())
    accuracies = [resultados[config]['accuracy'] for config in configuraciones]
    tiempos = [resultados[config]['time'] for config in configuraciones]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(configuraciones))
    bar_width = 0.4

    # Gráfica de precisión
    ax1.bar(indices - bar_width / 2, accuracies, bar_width, label='Precisión (%)', color='tab:blue')
    ax1.set_xlabel(f"Configuraciones de {parametro}", fontsize=12)
    ax1.set_ylabel("Precisión (%)", fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(configuraciones, rotation=45, fontsize=10)

    # Gráfica de tiempo de entrenamiento (segundo eje Y)
    ax2 = ax1.twinx()
    ax2.bar(indices + bar_width / 2, tiempos, bar_width, label='Tiempo (s)', color='tab:orange')
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=12)

    # Título y leyendas
    plt.title(f"Comparación de Resultados: {parametro}", fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

def entrenarConEpocasT_K(X_train, Y_train, epocas_lista, modelo_base=None, repeticiones=3, batch_size=32):
    """
    Entrena el modelo con diferentes números de épocas (con repeticiones), usando un modelo predefinido.
    """
    print("[Entrenar con Épocas] Ajustando número de épocas...")
    resultados = []
    tiempos = []  # Lista para almacenar los tiempos de entrenamiento
    
    # Verificamos si se pasó un modelo base
    if modelo_base is None:
        raise ValueError("Se debe pasar un modelo base para la experimentación.")
    
    for epocas in epocas_lista:
        print(f"[Entrenar con Épocas] Entrenando con {epocas} épocas...")
        historiales = []
        tiempos_epocas = []  # Lista para almacenar los tiempos de cada repetición
        
        for _ in range(repeticiones):
            # Usamos el modelo base proporcionado, y solo cambiamos el número de épocas
            model = modelo_base
            # Entrenamos el modelo con el número de épocas actual y el tamaño de lote
            history, training_time = entrenarMLP(model, X_train, Y_train, epocas=epocas, batch_size=batch_size)
            tiempos_epocas.append(training_time)  # Almacenamos el tiempo de cada repetición
            historiales.append(history)
        
        # Promediamos los resultados de las repeticiones
        resultados.append((epocas, historiales))
        tiempos.append(np.mean(tiempos_epocas))  # Promediamos el tiempo para el número de épocas actual
    
    print("[Entrenar con Épocas] Ajuste completado.")
    return resultados, tiempos


def entrenarConBatchSizeT_K(X_train, Y_train, batch_sizes, modelo_base=None, epocas=10, repeticiones=3):
    """
    Entrena el modelo con diferentes tamaños de lote (con repeticiones) usando un modelo preentrenado.
    """
    print("[Entrenar con Tamaño de Lote] Ajustando batch_size...")
    resultados = []
    tiempos = []  # Lista para almacenar los tiempos de entrenamiento
    
    if modelo_base is None:
        raise ValueError("Se debe pasar un modelo base para la experimentación.")
    
    for batch_size in batch_sizes:
        print(f"[Entrenar con Tamaño de Lote] Entrenando con batch_size={batch_size}...")
        historiales = []
        tiempos_batch = []  # Lista para almacenar los tiempos de cada repetición
        
        for _ in range(repeticiones):
            # Usamos el modelo base proporcionado, y solo cambiamos el tamaño de lote
            model = modelo_base
            # Entrenamos el modelo con el batch_size actual
            history, training_time = entrenarMLP(model, X_train, Y_train, epocas=epocas, batch_size=batch_size)
            tiempos_batch.append(training_time)  # Almacenamos el tiempo de cada repetición
            historiales.append(history)
        
        # Promediamos los resultados de las repeticiones
        resultados.append((batch_size, historiales))
        tiempos.append(np.mean(tiempos_batch))  # Promediamos el tiempo para el batch_size actual
    
    print("[Entrenar con Tamaño de Lote] Ajuste completado.")
    return resultados, tiempos



# =========================
# Código principal
# =========================




if __name__ == "__main__":
    # Cargar datos
    X_train, Y_train, X_test, Y_test, Y_train_cat, Y_test_cat = cargarCifar10()

    # Mostrar imágenes aleatorias
    print("\n[Principal] Mostrando imágenes aleatorias...")
    for i in sample(range(len(X_train)), 3):
        titulo = f"Clase: {Y_train[i][0]}"
        show_image(X_train[i], titulo)

    categorias = ["airplane", "automobile", "bird", "cat", "deer", 
              "dog", "frog", "horse", "ship", "truck"]

    # Tarea A: Entrenar un MLP básico
    print("\n=== Tarea A ===")
    mlp_model = crearMLP()
    history = entrenarMLP(mlp_model, X_train, Y_train_cat, epocas=10, batch_size=32)[0]
    graficarEvolucion(history.history, title="MLP Básico")
    evaluarMLP(mlp_model, X_test, Y_test_cat, labels=categorias)

    """

    # Tarea B: Ajustar el número de épocas
    print("\n=== Tarea B ===")
    epocas_lista = [5, 10, 20]
    resultados_epocas = entrenarConEpocas(X_train, Y_train_cat, epocas_lista)
    for epocas, historiales in resultados_epocas:
        average_history = {
            'loss': np.mean([h.history['loss'] for h in historiales], axis=0),
            'val_loss': np.mean([h.history['val_loss'] for h in historiales], axis=0),
            'accuracy': np.mean([h.history['accuracy'] for h in historiales], axis=0),
            'val_accuracy': np.mean([h.history['val_accuracy'] for h in historiales], axis=0)
        }
        graficarEvolucion(average_history, title=f"Épocas: {epocas}")
        # Generar matriz de confusión promedio para este número de épocas
        print(f"\nMatriz de Confusión Promedio para {epocas} épocas:")
        all_predictions = []  # Para almacenar todas las predicciones de las repeticiones

        for _ in range(len(historiales)):  # Iterar sobre las repeticiones realizadas
            model = crearMLP()  # Crear un nuevo modelo
            model.fit(X_train, Y_train_cat, epochs=epocas, batch_size=32, verbose=0)  # Entrenar
            predictions = model.predict(X_test).argmax(axis=1)  # Obtener predicciones
            all_predictions.append(predictions)  # Almacenar las predicciones
        
        # Llamar al método PromedioMatrizConfusion con las predicciones acumuladas
        PromedioMatrizConfusion(Y_test_cat, all_predictions, labels=categorias)

    

    # Tarea C: Ajustar el tamaño de lote
    print("\n=== Tarea C ===")
    batch_sizes = [16, 32, 64, 128]
    resultados_batch, tiempos_batch = entrenarConBatchSize(X_train, Y_train_cat, batch_sizes, 5)
    test_accuracies = [0.2, 0.3, 0.4, 0.5]

    # Llamar al método auxiliar para graficar la comparación
    print("[Principal] Graficando tiempo de entrenamiento por tamaño de lote...")
    graficarComparacionBatchSize(batch_sizes, tiempos_batch, test_accuracies)

    # Para las matrices de confusión, similar al ejemplo en Tarea B:
    all_predictions = []  # Para almacenar todas las predicciones de los diferentes batch sizes

    # Para cada batch_size, obtener las predicciones de todas sus repeticiones
    for batch_size, historiales in resultados_batch:
        print(f"\nMatriz de Confusión Promedio para batch size {batch_size}:")
        all_predictions = []  # Para almacenar todas las predicciones de las repeticiones
        
        for _ in range(len(historiales)):  # Iterar sobre las repeticiones realizadas
            model = crearMLP()  # Crear un nuevo modelo
            model.fit(X_train, Y_train_cat, epochs=10, batch_size=batch_size, verbose=0)  # Entrenar
            predictions = model.predict(X_test).argmax(axis=1)  # Obtener predicciones
            all_predictions.append(predictions)  # Almacenar las predicciones
    
        # Llamar al método PromedioMatrizConfusion con las predicciones acumuladas
        PromedioMatrizConfusion(Y_test_cat, all_predictions, labels=categorias)

        

    # Tarea D: Ajustar funciones de activación
    print("\n=== Tarea D ===")
    funciones_activacion = ['relu', 'sigmoid', 'tanh', 'elu']
    resultados_activaciones = probarActivaciones(X_train, Y_train_cat, X_test, Y_test_cat, funciones_activacion,labels=categorias)
    graficarResultadosActivaciones(resultados_activaciones)
    
    

    # Tarea E: Ajustar número de neuronas
    print("\n=== Tarea E ===")
    num_neuronas_lista = [16, 32, 64, 128]
    resultados_neuronas = probarNeurona(X_train, Y_train_cat, X_test, Y_test_cat, num_neuronas_lista, labels = categorias)
    graficarResultadosNeuronas(resultados_neuronas)

    

    # Tarea F: Optimizar un MLP con múltiples capas
    print("\n=== Tarea F ===")
    configuraciones = [[32], [64], [128, 64], [128, 64, 32]]
    resultados_capas = probarCapasOcultas(X_train, Y_train_cat, X_test, Y_test_cat, configuraciones, labels=categorias)
    graficarResultadosCapas(resultados_capas)

   
    

   # Tarea G: CNN sencilla con Keras
    print("\n=== Tarea G ===")
    # Llamamos a la función para comparar los modelos y obtener los resultados
    modelos, tiempos_batch, test_accuracies = compararModelos(X_train, Y_train_cat, X_test, Y_test_cat, epocas=10, labels=categorias)

    # Llamar a la función para graficar la comparación
    graficarComparativa(modelos, tiempos_batch, test_accuracies)


    """

    # Llamamos a la función para comparar los modelos con diferentes tamaños de filtro
    print("\n=== Tarea H ===")
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    resultados_filtros, all_true_labels, all_predictions = probarTamanosFiltro(X_train, Y_train_cat, X_test, Y_test_cat, kernel_sizes)

    # Aquí, puedes hacer la llamada a matrizConfusion para las predicciones de cada modelo.
    for i, kernel_size in enumerate(kernel_sizes):
        print(f"\n[Matriz de Confusión] Modelo con kernel_size={kernel_size}...")
        matrizConfusion(all_true_labels[i], all_predictions[i], labels=categorias)

    # Llamar a la función para graficar los resultados
    graficarResultadosKernel(resultados_filtros)


    

    # Tarea I: Optimizar arquitectura
    print("\n=== Tarea I ===")
    configuraciones_cnn = [[16, 32], [32, 64], [64, 64, 32]]
    resultados_cnn = optimizarCNN(X_train, Y_train_cat, X_test, Y_test_cat, configuraciones_cnn,labels=categorias)
    graficarResultadosCapas(resultados_cnn)

    
    

    # Tarea J: Creación y evaluación del conjunto de prueba propio
    print("\n=== Tarea J ===")
    X_custom, Y_custom = cargarImagenesPropias(15)

    

    # Evaluar la generalización con un modelo existente (ejemplo usando el modelo MLP)
    mlp_model = crearMLP()
    mlp_model.fit(X_train, Y_train_cat, epochs=10, batch_size=32, verbose=0)
    evaluarGeneralizacion(mlp_model, X_custom, Y_custom, labels=categorias)

    

    # Tarea K: Resultados y experimentación
    print("\n=== Tarea K ===")

    # Convertir Y_custom a one-hot encoding
    Y_custom_cat = to_categorical(Y_custom, num_classes=10)

    """
    
    # Experimentar con configuraciones de capas ocultas
    print("\n[Tarea K] Experimentando con Capas Ocultas...")
    configuraciones_capas = [[32], [64], [128, 64], [128, 64, 32]]
    resultados_capas = probarCapasOcultas(X_train, Y_train_cat, X_custom, Y_custom_cat, configuraciones_capas, labels=categorias)
    graficarResultadosCapas(resultados_capas)


    # Experimentar con funciones de activación
    print("\n[Tarea K] Experimentando con Funciones de Activación...")
    funciones_activacion = ['relu', 'sigmoid', 'tanh', 'elu']
    resultados_activacion = probarActivaciones(X_train, Y_train_cat, X_custom, Y_custom_cat, funciones_activacion, labels=categorias)
    graficarResultadosActivaciones(resultados_activacion)

    

    # Experimentar con optimizadores
    print("\n[Tarea K] Experimentando con Optimizadores...")
    optimizadores = {
        'Adam': Adam(),
        'SGD': SGD(learning_rate=0.01),
        'RMSprop': RMSprop(learning_rate=0.001)
    }
    resultados_optimizadores = experimentarOptimizadores(X_train, Y_train_cat, X_custom, Y_custom_cat, categorias, optimizadores, labels=categorias)
    graficarComparacionResultados(resultados_optimizadores, "Optimizadores")

    

    #Experimentar con diferentes epocas
    print("\n[Tarea K] Experimentando con diferentes epocas")
    epocas_lista = [5, 10, 20]
    resultados_epocas = entrenarConEpocas(X_train, Y_train_cat, epocas_lista)

    for epocas, historiales in resultados_epocas:
        average_history = {
            'loss': np.mean([h.history['loss'] for h in historiales], axis=0),
            'val_loss': np.mean([h.history['val_loss'] for h in historiales], axis=0),
            'accuracy': np.mean([h.history['accuracy'] for h in historiales], axis=0),
            'val_accuracy': np.mean([h.history['val_accuracy'] for h in historiales], axis=0)
        }
        graficarEvolucion(average_history, title=f"Épocas: {epocas}")
        
        # Generar matriz de confusión promedio para este número de épocas
        print(f"\nMatriz de Confusión Promedio para {epocas} épocas:")

        # Almacenamos las predicciones de todas las repeticiones
        all_predictions = [] 

        for _ in range(len(historiales)):  # Iterar sobre las repeticiones realizadas
            model = crearMLP()  # Crear un nuevo modelo
            model.fit(X_train, Y_train_cat, epochs=epocas, batch_size=32, verbose=0)  # Entrenar el modelo
            predictions = model.predict(X_custom).argmax(axis=1)  # Obtener predicciones sobre X_custom
            all_predictions.append(predictions)  # Almacenar las predicciones

        # Llamar al método PromedioMatrizConfusion con las predicciones acumuladas
        PromedioMatrizConfusion(Y_custom_cat, all_predictions, labels=categorias)

    """


    # Experimentar con tamaños del batch
    print("\n[Tarea K] Experimentando con diferentes batch_size")
    batch_sizes = [16, 32, 64, 128]
    resultados_batch, tiempos_batch = entrenarConBatchSizeT_K(X_train, Y_train_cat, batch_sizes, modelo_base=mlp_model)
    test_accuracies = [0.2, 0.3, 0.4, 0.5]
    graficarComparacionBatchSize(batch_sizes, tiempos_batch, test_accuracies)

    

    # Para las matrices de confusión, similar al ejemplo en Tarea B:
    all_predictions = []  # Para almacenar todas las predicciones de los diferentes batch sizes

    # Para cada batch_size, obtener las predicciones de todas sus repeticiones
    for batch_size, historiales in resultados_batch:

        print(f"\nMatriz de Confusión Promedio para batch size {batch_size}:")

        all_predictions = []  # Para almacenar todas las predicciones de las repeticiones
        
        for _ in range(len(historiales)):  # Iterar sobre las repeticiones realizadas
            model = crearMLP()  # Crear un nuevo modelo
            model.fit(X_train, Y_train_cat, epochs=10, batch_size=batch_size, verbose=0)  # Entrenar
            predictions = model.predict(X_custom).argmax(axis=1)  # Obtener predicciones
            all_predictions.append(predictions)  # Almacenar las predicciones
    
        # Llamar al método PromedioMatrizConfusion con las predicciones acumuladas
        PromedioMatrizConfusion(Y_custom_cat, all_predictions, labels=categorias)

    # Llamamos a la función para comparar los modelos con diferentes tamaños de filtro
    print("\n[Tarea K] Experimentando con diferentes tamanos de filtro")
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    resultados_filtros, all_true_labels, all_predictions = probarTamanosFiltro(X_train, Y_train_cat, X_custom, Y_custom_cat, kernel_sizes, labels=categorias)

    # Aquí, puedes hacer la llamada a matrizConfusion para las predicciones de cada modelo.
    for i, kernel_size in enumerate(kernel_sizes):
        print(f"\n[Matriz de Confusión] Modelo con kernel_size={kernel_size}...")
        matrizConfusion(all_true_labels[i], all_predictions[i], labels=categorias)

    # Llamar a la función para graficar los resultados
    graficarResultadosKernel(resultados_filtros)
