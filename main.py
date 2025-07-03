import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import matplotlib.pyplot as plt

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

# --------------------------
# Get class names
# --------------------------
class_names = train_ds.class_names
print("Detected classes:", class_names)

# --------------------------
# Performance optimizations
# --------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------
# Data augmentation layer
# --------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --------------------------
# Build the model
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
    layers.Dense(len(class_names))  # output layer with number of classes
])

# --------------------------
# Compile the model
# --------------------------
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# --------------------------
# Train the model
# --------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --------------------------
# Save the model
# --------------------------
os.makedirs("model", exist_ok=True)
model.save("model/car_damage_classifier.keras")
print("✅ Model saved to: model/car_damage_classifier.h5")

# --------------------------
# Plot Accuracy and Loss
# --------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
