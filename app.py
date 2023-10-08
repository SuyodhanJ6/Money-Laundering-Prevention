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

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

    

@app.get("/predict")
async def predictRouteClient():
    try:
        prediction_pipeline = PredictionPipeline()

        prediction_pipeline.initiate_prediction()

        return Response(
            "Prediction successful and predictions are stored in s3 bucket !!"
        )
    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    app_run(app=app, host= APP_HOST, port= APP_PORT)
    
