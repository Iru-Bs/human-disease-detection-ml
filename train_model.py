# ============================================================
# 🧠 HUMAN DISEASE DETECTION - FINE-TUNED MODEL
# ============================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# 🔹 Path to your dataset folder
DATASET_DIR = r"C:\Users\irapp\OneDrive\Pictures\Desktop\disease_dataset"

# 🔹 Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25  # you can increase to 30 for better accuracy

# ============================================================
# DATA GENERATOR
# ============================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("[INFO] Classes:", train_gen.class_indices)

# ============================================================
# MODEL
# ============================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 🔓 Unfreeze last few layers for fine-tuning
for layer in base_model.layers[-40:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# TRAIN
# ============================================================
print("[INFO] Training started...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ============================================================
# SAVE MODEL
# ============================================================
MODEL_PATH = r"C:\Users\irapp\OneDrive\Pictures\Desktop\disease_predictor_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Classes: {train_gen.class_indices}")

