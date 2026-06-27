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

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    result = predict(image)

    return result

    