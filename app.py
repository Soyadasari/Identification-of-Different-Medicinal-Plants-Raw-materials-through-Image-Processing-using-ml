from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import requests
import keras

app = Flask(__name__)

# Load pre-trained model (update this based on your model)
model = tf.keras.models.load_model('plant_identification_model.h5')

# Plant.id API Identification
def identify_plant(image_path, api_key):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    url = "https://api.plant.id/v2/identify"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": api_key
    }
    payload = {
        "images": [encoded_image],
        "organs": ["leaf"],
        "similar_images": False
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    try:
        plant_name = result['suggestions'][0]['plant_name']
        probability = result['suggestions'][0]['probability']
        return f"Prediction: {plant_name}\nConfidence: {probability:.2%}"
    except:
        return "Could not identify the plant."

# Preprocess the uploaded image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    uploaded_image = request.files['image']
    if uploaded_image.filename == '':
        return redirect(url_for('index'))

    try:
        # Verify it's an image file
        if not uploaded_image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('result.html', result="Please upload a valid image file (JPEG/PNG)")
        
        image = Image.open(uploaded_image)

        # Process image for model prediction
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)

        # Call the Plant.id API to identify the plant from the leaf image
        image_path = uploaded_image.filename  # You might want to save the file temporarily before passing the path
        api_key = "1xkwElNYMLOmnQPGnENd3zQyVkn6Eq2Ae9fQyVbH8uoo5Z5gbE"  # Add your Plant.id API key here
        plant_api_result = identify_plant(image_path, api_key)

        result_text = f"Model Prediction: {predictions}\nPlant API Prediction: {plant_api_result}"

        return render_template('result.html', result=result_text)

    except Exception as e:
        return render_template('result.html', result=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
