# Use the official Python base image
FROM continuumio/miniconda3

#LABEL maintainer="Swathi Thiruvengadam <thiruves@tf.uni-freiburg.de>"
WORKDIR /app

# Install make
RUN apt-get update && apt-get install -y make poppler-utils tesseract-ocr libgl1-mesa-glx libglib2.0-0 libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*

# Copy the environment file and the rest of the project into the container
COPY environment-thesis.yml .

# Install the dependencies
RUN conda env create -f environment-thesis.yml

# Activate the environment and install additional pip dependencies
RUN echo "source activate tables-detr-thesis" > ~/.bashrc

RUN /bin/bash -c "source activate tables-detr-thesis && pip install torch==2.1.2 torchvision==0.16.2"

# Create a requirements.txt file from the environment YAML
RUN conda run -n tables-detr-thesis pip freeze > requirements.txt

# Install pip dependencies
RUN conda run -n tables-detr-thesis pip install -r requirements.txt

# Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n", "tables-detr-thesis", "/bin/bash", "-c"]

# Copy the rest of your application code
COPY . .

# Ensure the Conda environment is activated by default
RUN echo "source activate tables-detr-thesis" >> ~/.bashrc

# Expose the port your application runs on
EXPOSE 8080

# Debugging: List installed packages in the environment
RUN conda run -n tables-detr-thesis python -c "import torch; print(torch.__version__)"

# Run the Makefile commands as needed
#CMD ["make", "inference-extraction"]
#CMD ["conda", "run", "--no-capture-output", "-n", "tables-detr-thesis", "make", "help"]
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tables-detr-thesis", "make", "help"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tables-detr-thesis"]
CMD ["conda", "run", "--no-capture-output", "-n", "tables-detr-thesis","make", "help"]
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tables-detr-thesis"]

# Docker build image
# docker build -t swathi-thiruvengadam-thesis .

# docker run container
# docker run -d --name tabular-data-extraction --shm-size=1g -p 8080:8080 swathi-thiruvengadam-thesis
# when facing dataloader issue increase shared memory --shm-size=1g 
