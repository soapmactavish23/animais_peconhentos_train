# type: ignore

import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Caminho para a pasta de imagens de treinamento
base_dir = './animals'

# Geração de dados e aumento de dados para prevenção de overfitting
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)  # 20% para validação

# Carregamento automático de imagens do diretório
train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset='validation')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes de cobras
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

def classify_snake(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    return class_indices[np.argmax(predictions)]


#Teste em cascavel
images = ['./animals/cascavel_teste.png', './animals/coral_teste.png', './animals/jararaca_teste.png', './animals/surucucu_teste.png']

# Classificação das novas imagens
for img_path in images:
    resultado_classificacao = classify_snake(img_path)
    print(f"Teste de imagem ({img_path}): {resultado_classificacao}")
