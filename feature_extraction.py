import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Feature Extraction Function using MobileNetV2 ---
def extract_deep_features(img_path, model, device, transform):
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t).cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f"Failed on {img_path}: {e}")
        return None

def main():
    DATASET_PATH = 'bears_dataset'

    # Use MobileNetV2 for lightweight feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model = base_model.features
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths = []
    labels = []
    for label in os.listdir(DATASET_PATH):
        label_folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_folder):
            continue
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            image_paths.append(img_path)
            labels.append(label)

    print(f"Total images found: {len(image_paths)}")

    features = []
    for img_path in tqdm(image_paths, desc="Extracting features"):
        feat = extract_deep_features(img_path, model, device, transform)
        if feat is not None:
            features.append(feat)

    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels[:len(X)])

    np.save('features.npy', X)
    np.save('labels.npy', y)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("Deep feature extraction complete. Files saved: features.npy, labels.npy, label_encoder.pkl")

if __name__ == '__main__':
    main()
