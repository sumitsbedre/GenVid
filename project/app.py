from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_path = "D:\sem\IBM experments\project\project\models\gender_classification_model.h5"
model = load_model(model_path)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_gender(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path.")
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_label = "Male" if prediction[0][0] > 0.5 else "Female"
    return class_label, prediction[0][0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)
            try:
                label, confidence = predict_gender(file_path)
                return render_template("result.html", label=label, confidence=confidence)
            except Exception as e:
                return str(e)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
