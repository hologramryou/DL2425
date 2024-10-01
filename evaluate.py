import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model (replace with your actual model path)
model = load_model('light_exported_mobilenetv2_model.h5')

# Class labels based on folder names
class_labels = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}

# Load and preprocess a new image for prediction (replace with the actual image path)
img_path = 'phoco.png'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input

# Make a prediction
predictions = model.predict(img_array)

# Get the predicted class index
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class index to the corresponding label
predicted_label = class_labels.get(predicted_class[0], 'Unknown')

# Output the predicted label
print(f'Predicted geographical category: {predicted_label}')
