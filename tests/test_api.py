from fastapi.testclient import TestClient
from unittest.mock import patch
from PIL import Image
import io

from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AQI Detection API is running!"
    }


@patch("app.predict")
def test_predict(mock_predict):

    # Fake prediction returned instead of the real model
    mock_predict.return_value = {
        "prediction": "Mock AQI",
        "confidence": 99.9
    }

    # Create a tiny fake image in memory
    image = Image.new("RGB", (224, 224), color="white")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": ("test.png", image_bytes, "image/png")
        }
    )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": "Mock AQI",
        "confidence": 99.9
    }

    # Verify our API actually called predict() exactly once
    mock_predict.assert_called_once()

def test_predict_rejects_non_image():

    response = client.post(
        "/predict",
        files={
            "file": ("test.txt", b"Hello World", "text/plain")
        }
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Please upload an image."
    }