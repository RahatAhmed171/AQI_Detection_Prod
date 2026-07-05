from config import (
    MODEL_REPO,
     MODEL_FILES,
    CLASS_NAMES
)

import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

from model import AQIClassifier

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def load_model(version: str):

    filename = MODEL_FILES[version]

    MODEL_PATH = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename
    )

    model = AQIClassifier(
        num_classes=6,
        patch_size=14
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model



def predict(image: Image.Image,version: str):
    image = val_test_transform(image)
    image = image.unsqueeze(0)
    model = load_model(version)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        predicted = probabilities.argmax(dim=1).item()

        confidence = probabilities.max().item()

    return {
    "model_version": version,
    "prediction": CLASS_NAMES[predicted],
    "confidence": round(confidence * 100, 2)
     }



