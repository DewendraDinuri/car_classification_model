import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("model/car_damage_classifier_YYYYMMDD-HHMMSS.keras")  # replace with your saved model name

# Define class names
class_names = ['01-minor', '02-moderate', '03-severe']

# Path to test image
img_path = 'test_images/car1.jpg'  # replace with your image path

# Preprocess the image
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # make it a batch

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
