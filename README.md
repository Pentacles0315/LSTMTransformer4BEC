# LSTMTransformer4BEC
## 1. models.py
The 'models.py' file contains all the architectures of deep learning models. In this file, different types of models are defined, LSTM, LSTM-Transformer and Bi-LSTM Transformer. Each model is encapsulated as a PyTorch nn Module class, convenient for importing and using in other files.
Main functions:
Define the structure of models
Provide reusable model architecture for different tasks
## 2. train.py
The 'train.py' file is responsible for the training process of the model. It imports the model architecture from models.py, loads data, and trains the model through a defined training loop. After training, the model will be saved to the specified path.
Main functions:
Import and initialize the model
Loading cleaned data
Define loss function and optimizer
Train the model and output the loss during the training process
Save the trained model
## 3. test.py
The 'test.py' file is used to evaluate and test the trained model. It loads the trained model, applies it to the test dataset, and calculates the performance metrics of the model, such as accuracy, precision, recall, etc.
Main functions:
Load the trained model
Loading test data
Evaluate the performance of the model and output the results
## 4. utils.py
Data preprocessing
...
