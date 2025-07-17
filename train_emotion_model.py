import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

print("=== Script started ===")

train_dir = "fer2013/train"
val_dir = "fer2013/test"

print("Train dir exists:", os.path.exists(train_dir))
print("Val dir exists:", os.path.exists(val_dir))

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical')

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical')

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=50, validation_data=val_data)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")
print("Model saved in models/emotion_model.h5")
