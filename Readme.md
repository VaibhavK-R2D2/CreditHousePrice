# Boston Housing Price Prediction

## Overview

This project implements a machine learning workflow for predicting house prices using the Boston Housing dataset. The workflow includes data loading, preprocessing, model training, evaluation, and model persistence.

## Models

* Decision Tree Regressor (`train.py`)
* Kernel Ridge Regressor (`train2.py`)

Reusable functions for preprocessing, training, prediction, and evaluation are defined in `misc.py`.

## Project Structure

```text
Boston_Housing/
│
├── data_loader.py
├── misc.py
├── train.py
├── train2.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Conda environment:

```bash
conda create -n digit python=3.11
conda activate digit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

Run the Decision Tree model:

```bash
python train.py
```

Run the Kernel Ridge model:

```bash
python train2.py
```

## Evaluation Metrics

The models are evaluated using:

* MAE
* MSE
* RMSE
* R² Score

## GitHub Actions

A GitHub Actions workflow is configured for the `kernelridge` branch. On every push, it automatically:

* Installs dependencies
* Runs `train.py`
* Runs `train2.py`
* Displays model performance

## Published By

Vaibhav Krishna G25AI1051
