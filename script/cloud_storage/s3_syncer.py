import os, sys

from script.logger import logging
from script.exception import MoneyLaunderingException



class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Uploads the contents of a local folder to an Amazon S3 bucket.
        
        :param folder: The local folder to upload.
        :param aws_bucket_url: The URL of the target Amazon S3 bucket.
        :raises MoneyLaunderingException: If an exception occurs during the process.
        """
        try:
            logging.info(f"Syncing folder '{folder}' to S3 bucket '{aws_bucket_url}'")
            
            # Construct the AWS CLI command
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            
            # Execute the AWS CLI command using os.system
            os.system(command)
            
            logging.info("Sync completed.")
        
        except Exception as e:
            # If an exception occurs, raise it with error details
            logging.error(f"Error during synchronization: {e}")
            raise MoneyLaunderingException(e, sys)


    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Downloads the contents of an Amazon S3 bucket to a local folder.
        
        :param folder: The local folder to download to.
        :param aws_bucket_url: The URL of the source Amazon S3 bucket.
        :raises MoneyLaunderingException: If an exception occurs during the process.
        """
        try:
            logging.info(f"Syncing folder from S3 bucket '{aws_bucket_url}' to '{folder}'")
            
            # Construct the AWS CLI command
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            
            # Execute the AWS CLI command using os.system
            os.system(command)
            
            logging.info("Sync completed.")
        
        except Exception as e:
            # If an exception occurs, raise it with error details
            logging.error(f"Error during synchronization: {e}")
            raise MoneyLaunderingException(e, sys)
