import numpy as np
import pandas as pd

insurance = pd.read_csv("insurance.csv")

# Closed Form Linear Regression (Part 2) -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Pre-processing and Encoding the data:
# Learned about the overall process from https://pbpython.com/categorical-encoding.html

# Shuffle dataset (as mentioned using a seed for reproducibility)
# ref -> https://stackoverflow.com/questions/71053540/how-to-shuffle-and-split-a-large-csv-with-headers
np.random.seed(0)
shuffled_data = insurance.sample(frac=1).reset_index(drop=True)

# Encode binary categorical variables using `replace` (since they are yes/no)
# ref -> https://stackoverflow.com/questions/3162614/python-search-and-replace-in-binary-file
shuffled_data['sex'] = shuffled_data['sex'].replace({'female': 0, 'male': 1})
shuffled_data['smoker'] = shuffled_data['smoker'].replace({'no': 0, 'yes': 1})

# One-hot encode 'region' categorical variable
# ref -> https://www.geeksforgeeks.org/ml-one-hot-encoding/
shuffled_data = pd.concat([shuffled_data, pd.get_dummies(shuffled_data['region'], prefix='region', dtype=int)], axis=1)

# Drop the original 'region' column
shuffled_data.drop('region', axis=1, inplace=True)

# Separate features and target
X = shuffled_data.drop('charges', axis=1)
y = shuffled_data['charges']

# Calculate the index at which to split the data (1/3 : 2/3 as mentioned in instructions)
split_index = int(len(shuffled_data) * (2 / 3))

# ref -> https://www.geeksforgeeks.org/how-to-split-data-into-training-and-testing-in-python-without-sklearn/
# Split the data into training and validation sets
training = shuffled_data[:split_index]
validate = shuffled_data[split_index:]

# Now split the training and validation data into X (features) and y (target)
X_train = training.drop('charges', axis=1)
y_train = training['charges']
X_validation = validate.drop('charges', axis=1)
y_validation = validate['charges']

# Add a bias column of ones to the features
# ref -> https://datascience.stackexchange.com/questions/24759/how-to-add-bias-consideration-into-logistic-regression-code
X_train_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train.values))
X_validation_with_bias = np.hstack((np.ones((X_validation.shape[0], 1)), X_validation.values))

# Calculate the weights using the pseudo-inverse (formula taken from lecture/Professor after class)
weights = np.linalg.pinv(X_train_with_bias.T @ X_train_with_bias) @ X_train_with_bias.T @ y_train

# Make predictions on the training and validation datasets
y_train_pred = X_train_with_bias.dot(weights)
y_validation_pred = X_validation_with_bias.dot(weights)

# Formulas taken from lecture notes
def RMSE(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def MSE(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def SMAPE(predictions, targets):
    return 100 / len(targets) * np.sum(2 * np.abs(predictions - targets) / (np.abs(targets) + np.abs(predictions)))

print("Training RMSE: ", RMSE(y_train_pred, y_train))
print("Validation RMSE: ", RMSE(y_validation_pred, y_validation))
print("Training SMAPE: ", SMAPE(y_train_pred, y_train))
print("Validation SMAPE: ", SMAPE(y_validation_pred, y_validation))
print()

# Cross-Validation (Part 3) -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def crossValidate(data, S):
    # np.random.seed(0)
    RMSE_values = []
    for run in range(20):
        # Each run will have a respective seed to randomize
        shuffled_data = data.sample(frac=1, random_state = run).reset_index(drop=True)

        # Encode binary categorical variables using `replace` (since they are yes/no)
        shuffled_data['sex'] = shuffled_data['sex'].replace({'female': 0, 'male': 1})
        shuffled_data['smoker'] = shuffled_data['smoker'].replace({'no': 0, 'yes': 1})

        # One-hot encode 'region' categorical variable
        # ref -> https://www.geeksforgeeks.org/ml-one-hot-encoding/
        shuffled_data = pd.concat([shuffled_data, pd.get_dummies(shuffled_data['region'], prefix='region', dtype=int)], axis=1)

        # Drop the original 'region' column
        shuffled_data.drop('region', axis=1, inplace=True)

        X = shuffled_data.drop('charges', axis=1)

        # Add a bias
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X.values))

        y = shuffled_data['charges']

        MSE_values = []

        for fold in range(S):

            # Calculate the fold size each time to handle cases where len(data) is not divisible by S
            fold_size_base = len(X_with_bias) // S
            rem = len(X_with_bias) % S # Extra 

            # If the fold index is less than the remainder, the fold will have an extra
            if fold < rem:
                fold_size = fold_size_base + 1
            else:
                fold_size = fold_size_base 
            
            # Indices upon what to split the data on
            start = fold * fold_size 
            end = start + fold_size

            # Split into training and validation for this fold
            # Ensure the training and validation data have numeric columns only
            # ref -> https://stackoverflow.com/questions/33705180/how-to-exclude-the-non-numerical-integers-from-a-data-frame-in-python
            # ref -> https://stackoverflow.com/questions/24641731/deleting-multiple-slices-from-a-numpy-array
            X_train_with_bias = np.delete(X_with_bias, slice(start, end), 0)
            y_train = np.delete(y.values, slice(start, end), 0)
            X_validation_with_bias = X_with_bias[start:end]
            y_validation = y.values[start:end]

            # Formula taken directly from lecture (as Professor also showed me after lecture)
            weights = np.linalg.pinv(X_train_with_bias.T @ X_train_with_bias) @ X_train_with_bias.T @ y_train
            y_validation_pred = X_validation_with_bias @ weights

            # Mean Squared Error
            MSE_validation = np.mean((y_validation - y_validation_pred)**2)

            MSE_values.append(MSE_validation)

        # Root Mean Squared Error
        RMSE_validation = np.sqrt(np.mean(MSE_values))
        
        RMSE_values.append(RMSE_validation)

        # print("X_train types:", type(X_train.dtypes))

    return [np.mean(RMSE_values), np.std(RMSE_values)]

# Load the insurance data
data = pd.read_csv('insurance.csv')

# Run crossValidate() for each S; returns RMSE followed by the STD
crossValidate3 = crossValidate(data, 3)
print(f"Cross-Validation results for S = 3: {crossValidate3}\n")

crossValidate223 = crossValidate(data, 223)
print(f"Cross-Validation results for S = 223: {crossValidate223}\n")

crossValidateN = crossValidate(data, len(data))
print(f"Cross-Validation results for {len(data)}: {crossValidateN}\n")
