import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = "ap-south-1"

