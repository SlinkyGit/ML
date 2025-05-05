import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logistic Regression
database = pd.read_csv("spambase.data")

# Seed the random number for reproducibility
np.random.seed(0)

shuffled_data = database.sample(frac=1).reset_index(drop=True)

# Calculate the index at which to split the data (1/3 : 2/3 as mentioned in instructions)
split_index = int(len(shuffled_data) * (2 / 3))

training = shuffled_data[:split_index]
validation = shuffled_data[split_index:]

def standardize(data, train_data):
    # Calculate mean and std using the training data
    # ref -> https://www.geeksforgeeks.org/create-the-mean-and-standard-deviation-of-the-data-of-a-pandas-series/
    mean = train_data.iloc[:, :-1].mean()
    std = train_data.iloc[:, :-1].std()
    
    # Apply the standardization to the given data
    data.iloc[:, :-1] = (data.iloc[:, :-1] - mean) / std
    
    return data

# Standardize training and validation data using training data mean and std
training_strd = standardize(training, training)
validation_strd = standardize(validation, validation)

# Sigmoid function (lecture slides - Logisitic Regression)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define feature and target matrices
# ref -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
X_train = training_strd.iloc[:, :-1].values
y_train = training_strd.iloc[:, -1].values
X_val = validation_strd.iloc[:, :-1].values
y_val = validation_strd.iloc[:, -1].values

# Add column of ones as bias
# Used in HW 1 and HW 2
X_train_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_validation_with_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))

# Initialize weights with random weights as seen in lecture slides (-/+ 10e-4)
# ref -> https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
# ref -> https://stackoverflow.com/questions/72025406/random-number-matrix-within-a-specified-range-in-python
theta = np.random.uniform(-0.0001, 0.0001, X_train_with_bias.shape[1])

def log_loss(y, y_hat):
    # Limit the values in array 
    ep = 1e-7 # Add episolon as professor mentioned in lecture, to account for log(0)
    return -np.mean(y * np.log(y_hat + ep) + (1 - y) * np.log(1 - y_hat + ep))

def logistic_regression(X_train, y_train, X_val, y_val, theta, learning_rate, epochs):
    m_train = len(y_train)
    log_loss_training = []
    log_loss_validation = []

    for _ in range(epochs):
        # Training predictions (calculating weights)
        z_train = np.dot(X_train, theta)
        y_hat_train = sigmoid(z_train) # Map onto [0, 1] using Sigmoid

        # Validation predictions (calculating weights)
        z_val = np.dot(X_val, theta)
        y_hat_val = sigmoid(z_val)
        
        # Derived from lecture notes: Batching formula for Log-Reg gradient
        gradient = np.dot(X_train.T, (y_hat_train - y_train)) / m_train
        
        # Update weights based on gradient
        theta -= learning_rate * gradient

        # Calculate log loss for training
        loss_train = log_loss(y_train, y_hat_train)
        log_loss_training.append(loss_train)

        # Calculate log loss for validation
        loss_val = log_loss(y_val, y_hat_val)
        log_loss_validation.append(loss_val)

    return theta, log_loss_training, log_loss_validation

# Set learning rate and epochs
learning_rate = 0.1
epochs = 1000

# Start training
theta, train_loss, val_loss = logistic_regression(X_train_with_bias, y_train, X_validation_with_bias, y_val, theta, learning_rate, epochs)

# To classify training/validation sample; spam if >= 50%
def classify(X, theta):
    return sigmoid(np.dot(X, theta)) >= 0.5

# Predict on training and validation sets
y_train_pred = classify(X_train_with_bias, theta)
y_val_pred = classify(X_validation_with_bias, theta)

# ref -> https://buildintelligence.medium.com/find-confusion-matrix-precision-recall-f1-score-and-accuracy-without-sklearn-27cad1bcbbea
# Computes performance metrics based on true and predicted binary classification labels
def compute_metrics(y_true, y_pred):

    # Initialize counters for true positives, true negatives, false positives, & false negatives
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # Iterate over each true and predicted label pair
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1

    # Calculate accuracy
    total_cases = len(y_true)
    if total_cases > 0:
        accuracy = (true_positives + true_negatives) / total_cases

    # Calculate precision
    positive_predictions = true_positives + false_positives
    if positive_predictions > 0:
        precision = true_positives / positive_predictions

    # Calculate recall
    actual_positives = true_positives + false_negatives
    if actual_positives > 0:
        recall = true_positives / actual_positives

    # Calculate F1 score
    precision_recall_sum = precision + recall
    if precision_recall_sum > 0:
        f1_score = 2 * (precision * recall) / precision_recall_sum

    return accuracy, precision, recall, f1_score

train_metrics = compute_metrics(y_train, y_train_pred)
val_metrics = compute_metrics(y_val, y_val_pred)

print("Training Metrics (Accuracy, Precision, Recall, F1):", train_metrics)
print("Validation Metrics (Accuracy, Precision, Recall, F1):", val_metrics)

# Plot Log-loss vs Epochs
def plot(log_loss_training, log_loss_validation):
    epochs = range(len(log_loss_training)) # Plot training epochs
    plt.figure(figsize=(15, 5)) 
    plt.plot(epochs, log_loss_training, label='Training Logarithmic-Loss')
    plt.plot(epochs, log_loss_validation, label='Validation Logarithmic-Loss')
    plt.title('Logarithmic-Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logarithmic-Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot(train_loss, val_loss)
# print(y_train_pred)
# print(y_val_pred)
