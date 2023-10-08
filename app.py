# import os, sys

# from script.logger import logging
# from script.exception import MoneyLaunderingException
# from script.pipeline import training_pipeline
from script.pipeline.training_pipeline import TrainPipeline
from script.constant.applicaton import APP_HOST, APP_PORT
from script.pipeline.prediction_pipeline import PredictionPipeline

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"message": "Training successful!"}
    except Exception as e:
        return {"error": f"Error occurred during training: {str(e)}"}

@app.get("/predict")
async def predictRouteClient():
    try:
        prediction_pipeline = PredictionPipeline()
        predicted_dataframe = prediction_pipeline.initiate_prediction()
        # You can return the prediction results as JSON or any other format
        return {"message": "Prediction successful!", "predictions": predicted_dataframe.to_dict()}
    except Exception as e:
        return {"error": f"Error occurred during prediction: {str(e)}"}

if __name__ == "__main__":
    app_run(app=app, host=APP_HOST, port=APP_PORT)