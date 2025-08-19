import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Original dataset-mapp
original_dir = "data"  # mappen som innehåller "cats" och "dogs"

# Skapa train/test mappar
base_dir = "cats_dogs_split"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

for directory in [train_dir, test_dir]:
    for category in ["cats", "dogs"]:
        os.makedirs(os.path.join(directory, category), exist_ok=True)

# Dela upp bilderna i train/test
split_ratio = 0.8

for category in ["cats", "dogs"]:
    files = os.listdir(os.path.join(original_dir, category))
    train_files, test_files = train_test_split(files, train_size=split_ratio, random_state=42)

    # Kopiera till train
    for f in train_files:
        src = os.path.join(original_dir, category, f)
        dst = os.path.join(train_dir, category, f)
        shutil.copyfile(src, dst)

    # Kopiera till test
    for f in test_files:
        src = os.path.join(original_dir, category, f)
        dst = os.path.join(test_dir, category, f)
        shutil.copyfile(src, dst)

# Parametrar
img_height, img_width = 128, 128
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Bygg CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Kompilera
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Träna modellen
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=15
)

# Utvärdera
val_loss, val_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {val_acc:.2f}")

# Plotta träningshistorik
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.show()
