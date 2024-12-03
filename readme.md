Repository Description
This repository contains the implementation of a hate speech classification pipeline, developed as part of the Headway Data Science Internship program at Giza Systems. The task focuses on classifying tweets as hate speech or not, using a labeled Twitter dataset. The pipeline leverages text preprocessing, feature engineering, sentiment analysis, and machine learning (ML) and deep learning (DL) models to achieve robust classification performance.

The project incorporates techniques to address data imbalance, improve interpretability, and enhance prediction accuracy through feature engineering and hyperparameter tuning.

Features
Data Loading and Preprocessing:

Ensures data integrity and prevents leakage through proper train-test splits.
Handles class imbalance with resampling methods.
Custom Text Preprocessor:

Cleans text using regex, tokenization, and lemmatization.
Implements TF-IDF vectorization for feature representation.
Sentiment Analysis and Feature Engineering:

Adds VADER sentiment scores to improve model representation.
Machine Learning Models:

Logistic Regression, SVM, Random Forest, and an optimized Voting Classifier.
Deep Learning Approach:

Feedforward neural network with dense layers and dropout regularization, fine-tuned using GridSearchCV.
Optimization:

Hyperparameter tuning using GridSearchCV.
Evaluation using Macro F1 score to address dataset imbalance.
File Structure
data/

Contains the dataset files (train.tsv, test.tsv).
notebooks/

Jupyter Notebooks for detailed experimentation and visualizations.
src/

Python modules for the pipeline, including:
text_preprocessing.py: Custom transformer for cleaning and vectorizing text.
sentiment_analysis.py: Module for computing VADER sentiment scores.
modeling.py: Scripts for training and evaluating models.
optimization.py: Hyperparameter tuning and ensemble methods.
results/

Evaluation results and model performance metrics.
README.md

Detailed project overview, setup instructions, and usage guidelines.
