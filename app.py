from logger import logger
import time
from fastapi import FastAPI, UploadFile, File, HTTPException,Query
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
async def predict_image(version: str = Query("v1"),file: UploadFile = File(...)):
    logger.info("Prediction request received")
    if version not in ["v1", "v2"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model version."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        start = time.perf_counter()
        result = predict(image,version)
        elapsed = time.perf_counter() - start
        logger.info(f"Prediction: {result['prediction']} | "f"Confidence: {result['confidence']}% | "f"Inference Time: {elapsed:.3f}s")
        return result

    except Exception:

        logger.exception("Prediction failed")

        raise

    