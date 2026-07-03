from logger import logger
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from inference import predict

app = FastAPI(
    title="AQI Image Classification API",
    description="Predict Air Quality Index category from an uploaded image",
    version="1.0.0"
)
@app.get("/")
def root():
    return {
        "message": "AQI Detection API is running!"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    logger.info("Prediction request received")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        start = time.perf_counter()
        result = predict(image)
        elapsed = time.perf_counter() - start
        logger.info(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}%")
        return result

    except Exception:

    logger.exception("Prediction failed")

    raise

    