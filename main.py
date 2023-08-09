import os, sys

from script.logger import logging
from script.exception import MoneyLaunderingException
from script.pipeline import training_pipeline
from script.pipeline.training_pipeline import TrainPipeline

from fastapi import FastAPI

if __name__ == '__main__':
    try:
        training_pieline = TrainPipeline()
        training_pieline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)