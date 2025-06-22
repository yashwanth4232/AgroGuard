# pred.py – Predict leaf disease from multiple images

import os
import logging
import warnings

# Suppress TensorFlow and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('crop_disease_model_improved.h5', compile=False)

# Class labels
class_names = ['Apple___Black_rot', 'Apple___healthy', 'Apple___rust', 'Apple___scab',
               'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato___Leaf_Mold']

# Treatment suggestions
treatment = {
    'Apple___Black_rot': 'Remove infected fruits, use fungicides like captan or myclobutanil.',
    'Apple___healthy': 'No action needed. Maintain proper watering and pruning.',
    'Apple___rust': 'Remove nearby juniper trees, apply fungicides early in season.',
    'Apple___scab': 'Prune trees, remove infected leaves, apply sulfur-based fungicides.',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Rotate crops, use resistant hybrids, apply fungicides.',
    'Corn___Common_rust': 'Use resistant hybrids and fungicides like azoxystrobin.',
    'Corn___healthy': 'No action needed.',
    'Corn___Northern_Leaf_Blight': 'Apply fungicides, plant resistant hybrids, rotate crops.',
    'Grape___Black_rot': 'Remove mummies, prune vines, apply myclobutanil fungicide.',
    'Grape___Esca_(Black_Measles)': 'Remove infected vines and use proper pruning techniques.',
    'Grape___healthy': 'No action needed.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply copper-based fungicides and remove debris.',
    'Potato___Early_blight': 'Use disease-free seed, apply fungicides like chlorothalonil.',
    'Potato___healthy': 'No action needed.',
    'Potato___Late_blight': 'Use resistant varieties, apply mancozeb or metalaxyl fungicide.',
    'Tomato___Leaf_Mold': 'Ensure ventilation, reduce humidity, use fungicides like mancozeb.'
}

# Folder containing test images
test_folder = 'test_leaf'
images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print("⚠️ No images found in 'test_leaf/'")
    exit()

print(f"\n🖼️ Found {len(images)} images in '{test_folder}/'\n")

# Predict each image
for idx, file in enumerate(images, 1):
    img_path = os.path.join(test_folder, file)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Simulate progress bar
    print(f"{idx}/{len(images)} ━━━━━━━━━━━━━━━━━━━━ 0s 161ms/step\n")

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    disease_name = class_names[class_index]
    confidence = np.max(prediction) * 100

    # Output
    print(f"📌 File: {file}")
    print(f"Disease: {disease_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Treatment: {treatment[disease_name]}\n")
