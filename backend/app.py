from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('path_to_trained_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
