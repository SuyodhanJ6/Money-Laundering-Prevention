import os
import boto3
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def read_s3_csv_to_dataframe(bucket_name, file_name):
    try:
        # Initialize a session using Amazon S3.
        session = boto3.Session(
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )

        # Initialize an S3 client using the session.
        s3 = session.client('s3')

        # Use the S3 client to download the file.
        response = s3.get_object(Bucket=bucket_name, Key=file_name)

        # Read the content of the file using Pandas.
        df = pd.read_csv(response['Body'])

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
