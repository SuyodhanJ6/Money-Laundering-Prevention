# Use the official Python 3.8.5 image as the base
FROM python:3.8.5-slim-buster

# Update the package list and install the AWS CLI
RUN apt update -y && apt install awscli -y

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory to the working directory in the container
COPY . /app

# Install the required Python packages from the requirements.txt file
RUN pip install -r requirements.txt

# Specify the command to run when the container starts
CMD ["python3", "app.py"]

# Specify the command to run when the container
