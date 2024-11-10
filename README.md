# Speech Emotion Recognition Using Keras, TensorFlow, and Scikit-Learn

This repository contains the code for a speech emotion recognition (SER) project that utilizes the Keras, TensorFlow, and Scikit-Learn libraries. This model is trained on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) to classify emotions conveyed through audio. It can detect various emotions from speech, providing valuable insights into the emotional state of speakers.

## Project Overview

Speech Emotion Recognition is a field focused on identifying emotions from audio data. This project aims to enable computers to recognize human emotions based on speech patterns, a capability that has applications in areas like healthcare, education, and entertainment.

Key emotions recognized by the model include neutral, calm, happy, sad, angry, fearful, disgust, and surprise.

## Installation

Ensure you have Python 3.8.10 installed. To install the necessary libraries, run:

```{Python}
pip install -r requirements.txt
```

## Data: RAVDESS Audio Dataset

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is a widely-used dataset for SER. It includes:

    7356 audio files performed by 24 actors (12 female, 12 male) with North American accents.
    Emotions: calm, happy, sad, angry, fearful, surprise, disgust (speech) and calm, happy, sad, angry, fearful (song).
    Intensity levels: normal and strong (excluding neutral).
    Audio format: 16-bit, 48kHz .wav.

## Data files

Place RAVDESS audio files in the input folder. Each audio file has a structured filename format indicating characteristics such as modality, emotion, intensity, and actor.

## Workflow and Folder Structure

The project is structured to follow a modular pipeline for easy understanding and scalability:

- input/: Contains the dataset and a config.ini file for customizable settings.
- src/: Contains the main scripts for model building and evaluation:
    - engine.py: Core script for model execution.
    - ml_pipeline/: Contains modular Python files for different functions.
    - utils.py: For data processing.
    - model.py: Model training script.
- output/: Stores the trained model for future use.
- lib/: Reference Jupyter notebook containing initial analysis and visualizations.

## Steps for Running the Project

1. Extract Features

To extract features from the audio files, run:

```python
python utils.py
```

2. Train the Model

Train the model using:

```python
python model.py
```

The model.py script in this repository is responsible for defining, training, and saving speech emotion recognition models. It supports two model architectures using different machine learning frameworks: Keras (TensorFlow) and Scikit-Learn. Based on configuration settings and the specified framework, this script handles model creation, training, and saving.
Script Overview

- Configuration:
    - Loads essential configuration settings from config.ini, such as model architecture (HIDDEN_LAYER_SHAPE), batch size (BATCH_SIZE), epochs (EPOCH), and input shape (INPUT_SHAPE).
    - Retrieves emotion labels and the subset of emotions to be learned from the configuration.

- Model Definitions:
    - get_model_keras: Defines a Keras neural network model with a customizable number of hidden layers. It uses ReLU activations for hidden layers and softmax for the output layer to classify emotions.
    - get_model_sklearn: Defines a Scikit-Learn Multi-Layer Perceptron (MLP) classifier with specified hidden layer sizes, batch size, and adaptive learning rate.

- Training Functions:
    - train_keras:
        - Splits data into training and testing sets.
        - Converts labels into categorical format for multi-class classification.
        - Defines the model using get_model_keras, then trains it with the Keras .fit() method.
        - Evaluates and prints the model’s performance on the test set.
        - Returns the trained Keras model.
    - train_sklearn:
        - Splits data into training and testing sets.
        - Defines the model using get_model_sklearn, then trains it with Scikit-Learn's .fit() method.
        - Tests the model and calculates accuracy using accuracy_score.
        - Returns the trained Scikit-Learn model.

- Training Wrapper:
    - train: A wrapper function that initiates model training based on the chosen framework (keras or sklearn).
    - Saves the trained model in the specified directory, either in Keras format (model.save()) or serialized with pickle for Scikit-Learn.
    - Returns the trained model and displays the save path.

- Model Retrieval:
    - get_model: Retrieves an untrained model based on the specified framework, useful for setting up model architecture without training.

3. Make Predictions

To classify emotions in a new audio sample, navigate to the src directory and execute:

```python
python engine.py --framework=(keras/sklearn) --infer --infer-file-path="path_to_audio_file"
```

## Technical Approach

- Feature Extraction: Audio features are extracted from .wav files using Librosa.
- Model Training:
    - Keras & TensorFlow: Used for building deep learning models.
    - Scikit-Learn: For creating a Multi-Layer Perceptron (MLP) classifier.
    - Hyperparameter Tuning: Optimizes model performance.
    - Modular Code Design: Allows for scalability and easy maintenance.

## Project Takeaways

- Basic neural network architecture and sound wave structure.
- Fourier Transform and sound wave visualization with spectrograms.
- Feature extraction with Librosa and data handling with Pandas and NumPy.
- Model training using Keras and Scikit-Learn.
- Hyperparameter tuning and saving trained models for reuse.

## Requirements

- Programming Language: Python 3.8.10
- Libraries:
    - Audio processing: librosa, soundfile
    - Data processing: pandas, numpy
    - Modeling: keras, tensorflow, sklearn
    - Visualization: matplotlib
    - Serialization: pickle
