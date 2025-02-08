from tensorflow.keras.models import load_model

model_path = "D:\sem\IBM experments\project\gender_classification_model.h5"  # Replace with your model path
model = load_model(model_path)

# Check the input shape
print(model.summary())
