# Sentiment Analysis Using LSTM

This repository contains a Python implementation of a sentiment analysis model using Long Short-Term Memory (LSTM) neural networks. The model is trained on a dataset of textual data labeled with sentiments (Positive and Negative) and is capable of classifying new texts based on their sentiment.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Make sure you have the following libraries installed:

- Python 3.x
- NumPy
- Pandas
- TensorFlow (Keras)
- Scikit-learn
- Regular expressions (`re`)

You can install the necessary packages using pip:

```bash
pip install numpy pandas tensorflow scikit-learn
```
# Dataset
The model uses a sentiment dataset named Sentiment.csv, which should contain at least two columns:
- text: The text content for sentiment analysis.
- sentiment: The sentiment label (Positive, Negative, Neutral).


The code filters out Neutral sentiments and processes the text for training.
# Installation
1. Clone this repository to your local machine:
   ``` bash
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   cd sentiment-analysis-lstm
    ```
2. Place the Sentiment.csv file in the root directory of the repository.

# Usage
1. Run the script:
     ```bash
     python sentiment_analysis.py
     ```
2. The model will output the accuracy and score of the evaluation, as well as the prediction for a sample tweet.
3. You can modify the twt variable in the script to test different sentences for sentiment classification.

# Model Architecture
The sentiment analysis model is built using the following layers:
1. Embedding Layer: Converts word indices to dense vectors of fixed size.
2. SpatialDropout1D Layer: Regularization layer to prevent overfitting.
3. LSTM Layer: Processes the sequences and captures the dependencies in the data.
4. Dense Layer: Outputs the sentiment classification (2 classes: Positive and Negative) using the softmax activation function.


# Results
After training the model, the code evaluates it on a test set, providing the accuracy for both Positive and Negative sentiments. The final prediction is displayed for the provided example tweet.

# Contributing
Feel free to contribute to this project by creating issues or submitting pull requests.

# License
This project is licensed under the MIT License. See the LICENSE file for more information.

``` typescript
You can save this content as `README.md` in your project directory. Let me know if you need any changes!
```

   
