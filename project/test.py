import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model_path = "Project\gender_classification_model.h5"  # Replace with your model file
model = load_model(model_path)

# Function to preprocess a single image
def preprocess_image(image_path):
    """Preprocess an image for model prediction."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Convert BGR to RGB (OpenCV loads images in BGR format by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the input size expected by the model (150x150)
    image = cv2.resize(image, (150, 150))

    # Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Expand dimensions to include batch size (1, 150, 150, 3)
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict and classify gender
def predict_gender(image_path):
    """Predict the gender from an image using the loaded model."""
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        class_label = "Male"
    else:
        class_label = "Female"

    return class_label, prediction[0][0]

# Test the model with a single image
test_image_path = "___"  # Replace with your test image file path
try:
    class_label, confidence = predict_gender(test_image_path)
    print(f"Predicted Class: {class_label} (Confidence: {confidence:.2f})")
except ValueError as e:
    print(e)

# If you want to test with a batch of images:
def predict_batch(image_paths):
    """Predict genders for a batch of images."""
    batch_images = []

    # Preprocess all images and stack them into a batch
    for img_path in image_paths:
        try:
            preprocessed_image = preprocess_image(img_path)
            batch_images.append(preprocessed_image[0])  # Remove batch dimension for stacking
        except ValueError as e:
            print(e)

    batch_images = np.array(batch_images)  # Convert list to numpy array

    # Predict for the batch
    predictions = model.predict(batch_images)
    
    # Interpret predictions
    results = []
    for i, prediction in enumerate(predictions):
        class_label = "Male" if prediction[0] > 0.5 else "Female"
        results.append((image_paths[i], class_label, prediction[0]))

    return results

# Test the model with a batch of images
# batch_image_paths = ["image1.jpg", "image2.jpg"]  # Replace with your test images
# batch_results = predict_batch(batch_image_paths)
# for path, label, confidence in batch_results:
#     print(f"Image: {path}, Predicted Class: {label}, Confidence: {confidence:.2f}")
