import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --------------------------
# Paths
# --------------------------
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'training')
val_dir = os.path.join(base_dir, 'validation')

# --------------------------
# Load datasets
# --------------------------
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(180, 180),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(180, 180),
    batch_size=32
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# --------------------------
# Performance optimizations
# --------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------
# Data augmentation
# --------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --------------------------
# Model
# --------------------------
model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))  # Output layer
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# --------------------------
# Early stopping
# --------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# --------------------------
# Train
# --------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop]
)

# --------------------------
# Save model with timestamp
# --------------------------
os.makedirs("model", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"model/car_damage_classifier_{timestamp}.keras"
model.save(model_path)
print(f" Model saved at: {model_path}")

# --------------------------
# Accuracy & loss plots
# --------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.tight_layout()
plt.show()

# --------------------------
# Evaluation: confusion matrix & report
# --------------------------
print("\n🔍 Evaluating on validation set...")

# Get true and predicted labels
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
