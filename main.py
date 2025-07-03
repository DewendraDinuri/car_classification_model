import tensorflow as tf #Main library for building and training deep learning models.
from tensorflow.keras.preprocessing import image_dataset_from_directory # automatically loads images from directories and prepares them for training.
import numpy as np # for numerical operations, especially with arrays.
import os # Used to work with directory paths (cross-platform).


# Paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'training') # Directory containing training images
val_dir = os.path.join(base_dir, 'validation') # Directory containing validation images

# Load datasets
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

# Class names
class_names = train_ds.class_names
print("Classes:", class_names)

# Performance config
AUTOTUNE = tf.data.AUTOTUNE # Automatically tune performance parameters  
#shuffle(1000): Randomizes the order of training samples to prevent overfitting.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # cache ()Stores data in memory after the first load for faster training.
#prefetch(): Lets the model train and load data at the same time, improving speed.
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))  # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save("model/car_damage_classifier.h5")
