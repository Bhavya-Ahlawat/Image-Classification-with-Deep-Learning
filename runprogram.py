import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
try:
    model = keras.models.load_model("cifar10_improved_model.keras")
except OSError:
    print("Error: 'cifar10_improved_model' not found. Make sure the model file exists in the correct directory or train the model first.")
    exit()  # Exit the script if the model isn't found

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(img_path):
    """Loads an image, preprocesses it, and makes a prediction using the loaded model."""
    try:
        img = image.load_img(img_path, target_size=(32, 32))
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    probability = np.max(predictions)  # Get the probability of the predicted class

    return predicted_label, probability

# Example usage (single image prediction):
# image_path = "path/to/your/image.jpg"  # Replace with the actual path to your image
# prediction_result = predict_image(image_path)

# if prediction_result:
#     predicted_label, probability = prediction_result
#     print(f"The predicted class for {image_path} is: {predicted_label} (Probability: {probability:.4f})")

def predict_images_from_directory(directory):
    """Predicts classes for all images in a given directory."""
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check for image file extensions
            filepath = os.path.join(directory, filename)
            prediction_result = predict_image(filepath)
            if prediction_result:
                predicted_label, probability = prediction_result
                results[filename] = (predicted_label, probability)
    return results

# Example usage (predicting multiple images from a directory):
image_directory = "./images"  # Replace with the path to your image directory
directory_predictions = predict_images_from_directory(image_directory)

if directory_predictions:
    for filename, (predicted_label, probability) in directory_predictions.items():
        print(f"Prediction for {filename}: {predicted_label} (Probability: {probability:.4f})")
else:
    print(f"No valid images found in the directory: {image_directory}")
