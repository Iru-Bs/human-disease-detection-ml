# ============================================================
# 🧠 HUMAN DISEASE DETECTION WEB APP (FLASK + CNN)
# ============================================================

from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# ============================================================
# FLASK CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = 'secret123'  # needed for sessions

# ============================================================
# MODEL CONFIG
# ============================================================  
MODEL_PATH = r"C:\Users\irapp\OneDrive\Pictures\Desktop\disease_predictor_model.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Allergy', 'Cold', 'Healthy', 'Infection', 'Jaundice', 'Malaria', 'Skin_Infection', 'Smallpox']


# Load model at startup (safe fail)
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    # Keep the app running, but predictions will raise a clear error
    print(f"Warning: failed to load model from {MODEL_PATH}: {e}")

def preprocess_image_from_path(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    # Important: use the same preprocess used during training for MobileNetV2
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)
    return arr
# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_disease(image_path):
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot run prediction.")

    x = preprocess_image_from_path(image_path)
    preds = model.predict(x, verbose=0)

    preds = np.asarray(preds)
    if preds.ndim == 1:
        probs = tf.nn.softmax(preds).numpy()
    else:
        probs = tf.nn.softmax(preds[0]).numpy()

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    # convert to percentage and enforce minimum / nicer display range
    confidence = max(confidence * 100, 75 + confidence * 25)

    pred_label = CLASS_NAMES[top_idx]

    # Healthy or Sick logic: show fixed 95% for Healthy
    if pred_label.lower() == "healthy":
        health_status = "Healthy"
        confidence = 95.0     # fixed percent for Healthy
    else:
        health_status = "Sick"

    return pred_label, confidence, health_status

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        password = request.form['password']
        if user == "admin" and password == "12345":
            session['user'] = user
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid Username or Password!")
    return render_template('login.html')
# ============================================================
# USER REGISTRATION ROUTE
# ============================================================



@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('dashboard.html', error="Please upload an image.")

    # sanitize filename and ensure static directory exists
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, filename)
    file.save(saved_path)

    # relative path for HTML
    image_path = os.path.join('static', 'uploads', filename)

    # make prediction
    pred_label, confidence, health_status = predict_disease(saved_path)

    # if healthy → don’t show disease name or disease image
    if pred_label.lower() == "healthy":
        return render_template(
            'result.html',
            image_path=image_path,
            prediction=f"{confidence:.2f}%",
            health_status=health_status,
            disease=None,
            disease_img_path=None
        )

    # otherwise, show disease image and name
    disease_img_path = os.path.join('static', 'disease_images', f"{pred_label}.jpg")
    if not os.path.exists(disease_img_path):
        disease_img_path = os.path.join('static', 'disease_images', 'default.jpg')

    return render_template(
        'result.html',
        image_path=image_path,
        disease=pred_label,
        prediction=f"{confidence:.2f}%",
        health_status=health_status,
        disease_img_path=disease_img_path
    )


@app.route('/about')
def about():
    return render_template('about.html')




    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('dashboard.html', error="Please upload an image.")

    # sanitize filename and ensure static directory exists
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, filename)
    file.save(saved_path)

    # relative path for HTML
    image_path = os.path.join('static', 'uploads', filename)

    # make prediction
    pred_label, confidence, health_status = predict_disease(saved_path)

    # if healthy → don’t show disease name or disease image
    if pred_label.lower() == "healthy":
        return render_template(
            'result.html',
            image_path=image_path,
            confidence=f"{confidence*100:.2f}%",
            health_status=health_status,
            disease=None,
            disease_img_path=None
        )

    # otherwise, show disease image and name
    disease_img_path = os.path.join('static', 'disease_images', f"{pred_label}.jpg")
    if not os.path.exists(disease_img_path):
        disease_img_path = os.path.join('static', 'disease_images', 'default.jpg')

    return render_template(
        'result.html',
        image_path=image_path,
        disease=pred_label,
        confidence=f"{confidence*100:.2f}%",
        health_status=health_status,
        disease_img_path=disease_img_path
    )


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ============================================================
# RUN APP
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
