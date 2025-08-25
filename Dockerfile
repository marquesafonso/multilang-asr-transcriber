# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# For local setup use:
# USER root
RUN useradd -m -u 1000 user

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY --chown=user . /app

#Install ImageMagick
RUN apt-get update && apt-get install -y imagemagick && sed -i '91d' /etc/ImageMagick-6/policy.xml
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 7860

# Run main.py when the container launches
CMD ["python", "main.py"]

USER 1000