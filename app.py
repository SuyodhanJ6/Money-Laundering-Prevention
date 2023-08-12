import os, sys

from script.logger import logging
from script.exception import MoneyLaunderingException
from script.pipeline import training_pipeline
from script.pipeline.training_pipeline import TrainPipeline
from script.constant.applicaton import APP_HOST, APP_PORT
from script.pipeline.prediction_pipeline import PredictionPipeline

from fastapi import FastAPI
from fastapi.responses import Response
from uvicorn import run as run_app



app = FastAPI()

@app.get("/train")
async def train_route():
    try:
        prediction_pipeline = PredictionPipeline()

        prediction_pipeline.initiate_prediction()

        return Response(
            "Prediction successful and predictions are stored in s3 bucket !!"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.get("/predict")
async def predict_route():
    try:
        pass

    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    run_app(app=app, host= APP_HOST, port= APP_PORT)
    
