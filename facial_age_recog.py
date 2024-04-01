# Importa las librerias
import tensorflow as tf
import inspect
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

# Inicialización y carga de datos
path = 'files/datasets/input/faces/'
directory = path + 'final_files/'
labels = pd.read_csv(path + "labels.csv")

# Crea un generador de imágenes
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen_flow = train_datagen.flow_from_dataframe(
    dataframe=labels,
    directory=directory,
    x_col='file_name',
    y_col='real_age',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    seed=12345)


# Obtiene un lote de imágenes y etiquetas
image_arrays, age_array = train_gen_flow.next()

# Información de las etiquetas
labels.info()

# Muestra los primeros registros
labels.head()

# Tamaño del conjunto de datos
print("Tamaño del conjunto de datos:", labels.shape)

# Descripción estadística de la edad
labels.describe()

# Explora la distribución de edad
plt.figure(figsize=(10, 6))
plt.hist(labels['real_age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Imprime algunas fotos para diferentes edades
unique_ages = labels['real_age'].unique()

# Selecciona algunas edades para mostrar
sample_ages = unique_ages[10:15]

plt.figure(figsize=(15, 8))
for i, age in enumerate(sample_ages):
    plt.subplot(2, 3, i+1)
    file = labels[labels['real_age'] == age]['file_name'].iloc[0]
    img = load_img('/datasets/faces/final_files/' + file)
    plt.imshow(img)
    plt.title(f'Edad: {age}')
    plt.axis('off')
plt.show()


def load_train(path):
    """
    Carga el conjunto de datos de entrenamiento.
    """
    train_datagen = ImageDataGenerator(validation_split=0.25,
                                       rescale=1.0/255
                                       )

    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,  # 32
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen_flow


def load_test(path):
    """
    Carga el conjunto de datos de prueba.
    """
    test_datagen = ImageDataGenerator(
        validation_split=0.25, rescale=1.0/255
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,  # 32
        class_mode='raw',
        subset='validation',
        seed=12345
    )

    return test_gen_flow


def create_model(input_shape):
    """
    Define el modelo utilizando la arquitectura ResNet50.
    """
    backbone = ResNet50(
        input_shape=input_shape,  # (224, 224, 3),
        weights='imagenet',
        include_top=False
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.0001),  # default is 0.001
        metrics=['mae']
    )

    print(model.summary())
    return model


def train_model(model,
                train_data,
                test_data,
                epochs=5,
                batch_size=None,
                steps_per_epoch=None,
                validation_steps=None):
    """
    Entrena el modelo utilizando los datos de entrenamiento y validación.
    """

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    """
    Entrena el modelo con los parametros dados
    """

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2
              )

    return model


# Sección de inicialización para el script
init_str = """
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

# Generar el script
with open('run_model_on_gpu.py', 'w') as f:
    f.write(init_str)
    f.write('\n\n')

    for fn_name in [load_train, load_test, create_model, train_model]:
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')
