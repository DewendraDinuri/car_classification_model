import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('model/car_damage_classifier.h5')

# Define class names (same order as training)
class_names = ['01-minor', '02-moderate', '03-severe']

# Load and preprocess image
img_path = 'test_images/sample.jpg'  # Replace with your test image path
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Make it batch of 1

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Output result
predicted_label = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"Predicted: {predicted_label} ({confidence:.2f}% confidence)")
