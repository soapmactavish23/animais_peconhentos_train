# Importando as bibliotecas necessárias
import numpy as np
import os
import cv2
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Definindo o caminho para as pastas com as imagens das cobras
base_dir = './animals'
categories = ['Cascavel', 'Coral', 'Jararaca', 'Surucucu']

# Carregando as imagens e as etiquetas
images = []
labels = []

for category in categories:
    path = os.path.join(base_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (300, 300))
        images.append(image)
        labels.append(category)

# Convertendo as listas em arrays do numpy
images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

# Binarizando as etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Dividindo os dados em conjuntos de treinamento e teste
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.2, random_state=42)

# Criando o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compilando o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10, verbose=1)

# Função para classificar uma imagem
def classify_snake(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300)).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    idx = np.argmax(prediction)
    return lb.classes_[idx]

# Testando o modelo com novas imagens
test_images = ['./animals/cascavel_teste.png', './animals/coral_teste.png', './animals/jararaca_teste.png', './animals/surucucu_teste.png']

for image_path in test_images:
    result = classify_snake(image_path)
    print(f"Teste de imagem ({image_path}): {result}")
