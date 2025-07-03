import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# Paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'training')
val_dir = os.path.join(base_dir, 'validation')

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
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
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
