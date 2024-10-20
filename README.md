
# Chatbot Using LSTM

This project implements a chatbot using Long Short-Term Memory (LSTM) networks, as seen in the `ModelRNN.ipynb` notebook file. The chatbot is designed to process sequences of questions and generate responses based on a dataset of question-answer pairs.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- Numpy
- Pandas
- Scikit-learn
- Jupyter Notebook (for running the notebook)

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy pandas scikit-learn jupyter
```

## Files Included

- **ModelRNN.ipynb**: The Jupyter notebook containing the code for training the chatbot.
- **extended_chat_data.csv**: (Expected file) A CSV containing questions and answers used for training.
- **chatBot.h5**: The saved trained chatbot model.

## Steps to Run

### 1. Open and Run the Jupyter Notebook

Launch Jupyter Notebook from the command line:

```bash
jupyter notebook
```

Open the `ModelRNN.ipynb` file and run the cells to train and evaluate the chatbot.

### 2. Data Preparation

In the notebook, data from `extended_chat_data.csv` is loaded, tokenized, and converted into padded sequences for training.

```python
data = pd.read_csv('extended_chat_data.csv', on_bad_lines='skip')
questions = data['Question'].values
answers = data['Answer'].values
```

### 3. Train the Model

The model consists of:

- **Embedding Layer**: Converts input sequences to dense vectors.
- **LSTM Layer**: Handles the sequential dependencies between words.
- **Dense Layer**: Outputs a probability distribution over the vocabulary.

The model is trained using `sparse_categorical_crossentropy` loss and Adam optimizer.

```python
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_seq_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))
```

### 4. Saving and Loading the Model

The trained model is saved as `chatBot.h5` and can be loaded for future use.

```python
model.save("chatBot.h5")
model = load_model("chatBot.h5")
```

### 5. Predicting Responses

The chatbot predicts responses by converting the input text to sequences, running it through the model, and then converting the predicted sequence back to text.

```python
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_len)
    prediction = model.predict(input_seq)
    response_seq = np.argmax(prediction, axis=-1)
    response = tokenizer.sequences_to_texts([response_seq])
    return response[0]
```

### 6. Example Interaction

```bash
You: Hello
Bot: Hi, how can I help you?
```

### 7. Usage

1. Download or clone this repository.
2. Place the `extended_chat_data.csv` file in the project directory.
3. Open the notebook and run the cells to train the model.

## License

This project is licensed under the MIT License.
