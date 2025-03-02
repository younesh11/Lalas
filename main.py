import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_inference import TextClassifier

app = FastAPI()

model_path = "./distilbert-finetuned"
classifier = TextClassifier(model_path)


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_class: int
    probabilities: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        # Perform inference
        predicted_class, probabilities = classifier.predict(request.text)
        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5005)