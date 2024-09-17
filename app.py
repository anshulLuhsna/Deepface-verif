from flask import Flask, render_template, request, jsonify, redirect, url_for
from deepface import DeepFace
import os
import numpy as np
from PIL import Image
import json
import cv2
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/verify', methods=['POST'])
def verify():
    if 'file1' not in request.files or 'file2' not in request.files:
        print("Both required")
        return jsonify({'error': 'Both files are required'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1 and file2:
        # Load the uploaded images using Pillow
        image1 = Image.open(file1).convert('RGB')
        image2 = Image.open(file2).convert('RGB')

        # Convert images to numpy arrays
        image1_array = np.array(image1)
        image2_array = np.array(image2)

        # Detect the face in image1 (assuming only one face)
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        targets = face_detector.detectMultiScale(image1_array, 1.3, 5)

        if len(targets) > 0:
            x, y, w, h = targets[0]
            target = image1_array[int(y):int(y+h), int(x):int(x+w)]

            # Detect faces in image2 (the one with multiple faces)
            faces = face_detector.detectMultiScale(image2_array, 1.3, 5)

            at_least_one_verified = False
            results = []
            for (x, y, w, h) in faces:
                face = image2_array[int(y):int(y+h), int(x):int(x+w)]
                try:
                    result = DeepFace.verify(target, face, model_name="VGG-Face", enforce_detection=False)
                    print(result)
                    is_verified = result['verified']
                    results.append({'verified': is_verified, 'distance': result['distance']})

                    # If at least one verification is true, set flag to true
                    if is_verified:
                        at_least_one_verified = True

                except ValueError as e:
                    results.append({'verified': False, 'distance': None, 'error': str(e)})

            # Return true if at least one verification is true, otherwise false
            return jsonify({'results': results, 'verified': at_least_one_verified})

        else:
            return jsonify({'error': 'No face detected in the first image'}), 400

    else:
        return jsonify({'error': 'Failed to process images'}), 500

if __name__ == '__main__':
    app.run(debug=True)
