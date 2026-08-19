from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os


app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered 38-class plant disease detection API",
    version="1.0.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODEL
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "plant_disease_final_96_89.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# 38 CLASSES
# =========================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "🌱 Plant Disease Detection API is running!",
        "status": "success",
        "classes": 38
    }


# =========================
# PREDICTION
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check image type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize exactly like model input
        image = image.resize((224, 224))

        # Convert to numpy
        image_array = np.array(image).astype("float32")

        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Top prediction
        predicted_index = int(np.argmax(predictions))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(predictions[predicted_index] * 100)

        # Top 5
        top_indices = np.argsort(predictions)[-5:][::-1]

        top5 = []

        for index in top_indices:
            top5.append({
                "class": CLASS_NAMES[int(index)],
                "confidence": round(
                    float(predictions[index] * 100),
                    2
                )
            })

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "top_5": top5
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )