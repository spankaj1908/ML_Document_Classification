# ML_Document_Classification
A small project to train a Neural Network on Documents data so that those documents can be classified into different categories.

## Folder Structure

The project has the following folders:

- Model: This folder contains the code for defining the machine learning model.

  - model.py: This file contains the code for defining the model architecture and other variables.

- Script: This folder contains the scripts for preparing the data, training the model, evaluating the model, and extracting features from the data.

  - processed_data.py: This file contains code for pre-processing the raw data imported from sklearn library
  - extract_features.py: This file contains code for extracting features from the data using the trained model.
  - model_training.py: This file contains code for training the machine learning model.
  - evaluate_model.py: This file contains code for evaluating the performance of the trained model.

- Notebook: This folder contains a Jupyter notebook that demonstrates the analysis done to come up with these scripts

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies (listed in `requirements.txt`).
3. Run the `processed_data.py` script to preprocess the raw data.
4. Run the `extract_features.py` script to extract features from the data
5. Run the `model_training.py` script to train the model.
6. Run the `evaluate_model.py` script to evaluate the performance of the trained model.

## Dependencies

This project requires the following dependencies:

- Python 3.x
- scikit-learn
- torch
- regex
- nltk
- jupyter

You can install the dependencies by running:

pip install -r requirements.txt
