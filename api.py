from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import os

app = FastAPI(title="SuperStore & AI Text API", version="1.0.0")

# Configure CORS for Next.js frontend (typically runs on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Variables ---
rf_model = None
nn_model = None
ai_text_nn = None
tfidf_vectorizer = None

# --- Application Startup Event ---
@app.on_event("startup")
async def load_models():
    global rf_model, nn_model, ai_text_nn, tfidf_vectorizer
    print("Loading models...")
    
    # Load RF
    try:
        rf_model = joblib.load("rf_model.pkl")
        print("Loaded Random Forest model.")
    except Exception as e:
        print(f"Error loading RF model: {e}")

    # Load NN
    try:
        model_path = "nn_model.keras" if os.path.exists("nn_model.keras") else "nn_model.h5"
        nn_model = tf.keras.models.load_model(model_path)
        print("Loaded Neural Network model.")
    except Exception as e:
        print(f"Error loading NN model: {e}")

    # Load AI Text NN
    try:
        ai_text_nn = tf.keras.models.load_model("ai_text_nn.keras")
        print("Loaded AI Text NN model.")
    except Exception as e:
        print(f"Error loading AI Text NN model: {e}")

    # Load TF-IDF
    try:
        tfidf_vectorizer = joblib.load("tfidf.pkl")
        print("Loaded TF-IDF Vectorizer.")
    except Exception as e:
        print(f"Error loading TF-IDF Vectorizer: {e}")


# --- Pydantic Models for Requests ---
class ProfitPredictionRequest(BaseModel):
    sales: float
    quantity: int
    discount: float
    shipping_cost: float
    category: int
    sub_category: int
    region: int
    segment: int

class AITextRequest(BaseModel):
    text: str


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running. Models loaded."}


@app.post("/predict/profit/rf")
def predict_profit_rf(request: ProfitPredictionRequest):
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Random Forest model is not available.")
    
    data = np.array([[
        request.sales, request.quantity, request.discount, request.shipping_cost,
        request.category, request.sub_category, request.region, request.segment
    ]])
    
    try:
        prediction = rf_model.predict(data)
        result = int(prediction[0])
        status = "Profit" if result == 1 else "Loss"
        return {"model": "Random Forest", "prediction": result, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/profit/nn")
def predict_profit_nn(request: ProfitPredictionRequest):
    if nn_model is None:
        raise HTTPException(status_code=503, detail="Neural Network model is not available.")
    
    data = np.array([[
        request.sales, request.quantity, request.discount, request.shipping_cost,
        request.category, request.sub_category, request.region, request.segment
    ]])
    
    try:
        prediction = nn_model.predict(data)
        probability = float(prediction[0][0])
        result = 1 if probability > 0.5 else 0
        status = "Profit" if result == 1 else "Loss"
        return {"model": "Neural Network", "prediction": result, "status": status, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ai-text")
def predict_ai_text(request: AITextRequest):
    if ai_text_nn is None or tfidf_vectorizer is None:
         raise HTTPException(status_code=503, detail="AI Text models are not fully loaded.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        # Preprocess
        vectorized_text = tfidf_vectorizer.transform([request.text]).toarray()
        
        # Predict
        prediction = ai_text_nn.predict(vectorized_text)
        probability = float(prediction[0][0])
        
        # Result analysis (Assuming >0.5 is AI)
        is_ai = probability > 0.5
        label = "AI" if is_ai else "Human"
        confidence = probability if is_ai else (1 - probability)

        return {
            "prediction_label": label,
            "is_ai": is_ai,
            "ai_probability": probability,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
