from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from machine_learning.titanic import PredictOnAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"], 
    allow_headers=["*"]  
)

class TitanicRequest(BaseModel):
    Sex: str      
    Pclass: str    
    Age: int      
    Parch: int     
    SibSp: int     

class TitanicResponse(BaseModel):
    survival_probability: float

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/api/titanic", response_model=TitanicResponse)
def predict_survival(request: TitanicRequest):
    try:
        survival_probability = PredictOnAPI.derive_survival_probability(
            Sex=request.Sex,
            Pclass=request.Pclass,
            Age=request.Age,
            Parch=request.Parch,
            SibSp=request.SibSp,
        )
        return {"survival_probability": survival_probability}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Model file not found. Please ensure the model is trained.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
