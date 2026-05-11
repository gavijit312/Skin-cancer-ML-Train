"""FastAPI inference server for skin cancer model.

Endpoints:
  GET  /health        -> Server status and model info
  POST /predict       -> Upload image, get classification

Usage:
  uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_api")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "GroundTruth.csv"
INPUT_SIZE = 224

# Global state
model = None
class_names = []


def load_class_names():
    """Load class names from GroundTruth.csv if it exists."""
    if not CSV_PATH.exists():
        return None
    
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            return [c for c in header if c != "image"]
    except Exception as e:
        logger.warning(f"Failed to load class names from CSV: {e}")
        return None


def preprocess_image(image_bytes: bytes, size: int = INPUT_SIZE) -> np.ndarray:
    """Load, resize, and preprocess image bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")
    
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


app = FastAPI(
    title="Skin Cancer Classification API",
    description="Inference server for EfficientNetB0 skin cancer classifier",
    version="1.0"
)


def custom_batch_normalization(**kwargs):
    """Custom BatchNormalization loader that strips unsupported kwargs from older TF versions."""
    # Remove renorm-related kwargs that Keras 3.x doesn't support
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    return tf.keras.layers.BatchNormalization(**kwargs)


@app.on_event("startup")
def load_model():
    """Load model and class names on startup."""
    global model, class_names
    
    logger.info(f"Loading model from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    try:
        # Use custom objects to handle BatchNormalization compatibility across TF versions
        custom_objects = {
            'BatchNormalization': custom_batch_normalization
        }
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False, custom_objects=custom_objects)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Load class names
    names = load_class_names()
    if names:
        class_names = names
        logger.info(f"Loaded {len(class_names)} class names: {class_names}")
    else:
        try:
            num_classes = model.output_shape[-1]
            class_names = [f"class_{i}" for i in range(num_classes)]
            logger.info(f"Generated {num_classes} default class names")
        except Exception:
            logger.warning("Could not determine number of classes")


@app.get("/health")
def health_check():
    """Check server health and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "class_names": class_names
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin cancer classification from uploaded image.
    
    Returns:
      - predicted_class: Class name with highest probability
      - confidence: Confidence score (0-1)
      - probabilities: Dict of all class probabilities
      - predicted_index: Index of predicted class
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (got {file.content_type})"
        )
    
    # Read and preprocess image
    contents = await file.read()
    try:
        img_array = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Run prediction
    try:
        preds = model.predict(img_array, verbose=0)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    # Parse results
    probs = preds[0].tolist()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_label = class_names[top_idx] if (class_names and top_idx < len(class_names)) else str(top_idx)
    
    # Build probability map
    prob_map = {}
    for i, p in enumerate(probs):
        label = class_names[i] if (class_names and i < len(class_names)) else str(i)
        prob_map[label] = float(p)
    
    return JSONResponse({
        "predicted_class": top_label,
        "predicted_index": top_idx,
        "confidence": top_prob,
        "probabilities": prob_map
    })


if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="127.0.0.1", port=8000)
