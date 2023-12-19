#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model 
from PIL import Image, ImageOps  
import wikipedia
from data import cook, FOOD_NAME

app = Flask(__name__)

BASE_URL = "127.0.0.1:5000"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
LABELS_PATH = os.path.join(APP_ROOT, 'models/labels.txt')
MODEL_PATH = os.path.join(APP_ROOT, 'models/best_model.h5')

app.config['UPLOAD'] = UPLOAD_FOLDER    

np.set_printoptions(suppress=True)

model = load_model(MODEL_PATH, compile=False)

class_names = open(LABELS_PATH, "r").readlines()

import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD'], filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def recognize():
    file = request.files.get('image')
    if file:
        filename = secure_filename(file.filename)
        file_name_random = get_random_string(12) + filename
        filepath = os.path.join(app.config['UPLOAD'], file_name_random)
        file.save(filepath)
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        image = Image.open(filepath).convert("RGB")

        size = (224, 224)
        
        image = ImageOps.fit(image, size, Image.BICUBIC)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        try:
            food_name = str(FOOD_NAME[int(class_name[2:]) - 1])
            summary = wikipedia.summary(food_name)
            
            return jsonify({
                'message': 'success',
                'class': str(class_name[2:]),
                'food_name': str(FOOD_NAME[int(class_name[2:]) - 1]),
                'confidence': str(int(float(confidence_score) * 100)) + "%",
                'image_path': "/uploads" + "/" + file_name_random,
                'description': summary,
                "cook": cook[int(class_name[2:]) - 1]
            })
        except:
            return jsonify({
                'message': 'success',
                'class': str(class_name[2:]),
                'food_name': str(FOOD_NAME[int(class_name[2:]) - 1]),
                'confidence': str(int(float(confidence_score) * 100)) + "%",
                'image_path': "/uploads" + "/" + file_name_random,
                'description': "The information about this dish cannot be found on wiki at the moment.",
                "cook": cook[int(class_name[2:]) - 1]
            })
    else:
        return jsonify({'message': 'No file uploaded!'})
    

if __name__ == "__main__":
    app.run(debug=True)

