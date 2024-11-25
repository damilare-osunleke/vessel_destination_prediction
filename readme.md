# Project Title: Prediction of Destination Country for Container Vessels   

## Brief Description:
 This project explores the application of a deep learning model to predict the next country of call for container ships, leveraging historical positional data and vessel characteristics. The implementation incorporates a novel method that transforms longitudinal and latitudinal data into h3 tiles, enhancing spatial information processing to improve destination prediction accuracy in maritime settings. The developed model employs Long Short-Term Memory (LSTM) networks to effectively capture the temporal dependencies of vessel movements, addressing a multiclass classification problem involving 175 different classes.

## Content:

**code:** this folder contains all the code for the project and the uder-defined helper libraries.
    - 1_data_preprocessing.py: preprocessing and cleaning of the data.
    - 2_data_exploration_visualisation.ipynb: a notebook for vatious visualisation of the the data and obtaining key statistics.
    - 3_model_training_optimisation.py: model training and optimisation.
    - 4_best_model_training.py: retraining of the best model obtained from 3 above to   obtain the final model.
    - 5_evaluation.py: evaluation of the best model performance.
    - lstm_model.py : contains the list model.
    - data_processing_helpers.py: contains various helper functions for data processing.
    - model_helpers.py: contains various helper functions for model training.
    - evaluation_helpers: contains helper functions for evaluation of the model.
    - visualisation_helpers: contains helper function for visualising trips on a folium map.


**temp:** this contains several intermediate outputs from one notebook which serve as inputs  for some other notebook(s)
**output:** Final outputs. Apart from general outputs, this folder has subfolders.
