from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- Configuration ---
UPLOAD_FOLDER = 'static'
CLASSES = ['F0', 'F1', 'F2', 'F3', 'F4']
MODEL_PATH = 'model/best_model.pth'  # adjust if needed

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Create Static Folder if not exists ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model ---
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# --- Predict Image ---
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return CLASSES[predicted.item()]

# Load model once globally
model = load_model(MODEL_PATH)

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            prediction = predict_image(image_path, model)

    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Liver Fibrosis Detection</title>
        </head>
        <body>
            <h2>Liver Fibrosis Stage Prediction</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <input type="submit" value="Predict">
            </form>
            
            {% if image_path %}
                <h3>Uploaded Image:</h3>
                <img src="{{ image_path }}" width="300">
            {% endif %}
            
            {% if prediction %}
                <h2>Predicted Stage: {{ prediction }}</h2>
            {% endif %}
        </body>
        </html>
    ''', prediction=prediction, image_path=image_path)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
