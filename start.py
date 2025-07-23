import numpy as np
import pickle
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import transforms
import os

# Load label encoder
le = pickle.load(open('label_encoder.pkl', 'rb'))
num_classes = len(le.classes_)

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load model
class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeatureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# Load model state
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features_dim = np.load('features.npy').shape[1]
model = FeatureClassifier(features_dim, num_classes)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from torchvision.models import mobilenet_v2
mobilenet = mobilenet_v2(pretrained=True).features.eval().to(device)

def extract_features_from_image(img_pil):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = mobilenet(img_tensor).mean([2, 3]).cpu().numpy().flatten()
    return features

def predict(image):
    features = extract_features_from_image(image)
    features_scaled = scaler.transform([features])
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(features_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Animal Species Classifier",
    description="Upload an image to classify animal species."
).launch()
