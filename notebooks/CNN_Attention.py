!pip install -q kaggle scikit-learn seaborn

from google.colab import files
import os

!rm -rf ~/.kaggle
!rm -f kaggle.json

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("\nPlease upload your kaggle.json file:")
    uploaded = files.upload()
    if 'kaggle.json' in uploaded:
        !mkdir -p ~/.kaggle
        !cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        print("\nKaggle API token configured.")
    else:
        raise Exception("Aborted: kaggle.json not uploaded.")
else:
    print("\nKaggle API token already configured.")

!kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification

!unzip -q gtzan-dataset-music-genre-classification.zip

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'Data/images_original'

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Genres found:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.1),
])
base_model = applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = applications.efficientnet.preprocess_input(x) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

phase1_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
initial_epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[phase1_early_stopping]
)

base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = len(base_model.layers) - 20 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning starting from layer {fine_tune_at} onwards.")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8, 
    verbose=1,
    restore_best_weights=True
)
print("\n--- PHASE 2: FINE-TUNING THE TOP LAYERS ---")
fine_tune_epochs = 50
total_epochs = len(history.epoch) + fine_tune_epochs

history_fine_tune = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_ds,
    callbacks=[fine_tune_early_stopping]
)

loss, accuracy = model.evaluate(val_ds)
print("\n-------------------------------------------------")
print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")
print("-------------------------------------------------")

y_pred_probs = []
y_true = []
for images, labels in val_ds:
    y_pred_probs.extend(model.predict(images, verbose=0))
    y_true.extend(labels.numpy())

y_pred = np.argmax(np.array(y_pred_probs), axis=1)
y_true = np.array(y_true)

print("\nGenerating Classification Report...")
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)
