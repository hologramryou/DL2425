import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Set up directories
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# Create data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=16,  # Reduced batch size for faster training steps
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Path to the downloaded weights file
local_weights_path = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'

# Load the base model with local weights
base_model = MobileNetV2(weights=local_weights_path, include_top=False)

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Reduced dense layer size to 512
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for fewer epochs
history = model.fit(
    train_generator,
    steps_per_epoch=100 // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=50 // validation_generator.batch_size,
    epochs=3  # Reduced epochs to 3
)

# Evaluate the model
score = model.evaluate(validation_generator)
print(f'Validation loss: {score[0]}')
print(f'Validation accuracy: {score[1]}')

# Save the trained model
model.save('light_exported_mobilenetv2_model')  # Save in TensorFlow's SavedModel format
# model.save('exported_mobilenetv2_model.h5')  # Alternatively, save in HDF5 format

# Load and preprocess a new remote sensing image
img_path = 'file.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
class_labels = train_generator.class_indices  # Get class labels
predicted_label = list(class_labels.keys())[predicted_class[0]]
print(f'Predicted geographical category: {predicted_label}')
