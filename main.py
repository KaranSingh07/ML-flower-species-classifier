import numpy as np
import pandas as pd   # np and pd are just python conventions for these two

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print(f"Keys of iris_dataset: \n{iris_dataset.keys()}")

short_desc = iris_dataset['DESCR'][:230]
print("Short Description: \n" + short_desc + "\n...")

# 3 classes of flowers
target_names = iris_dataset['target_names']
print(f"Target names (3 Classes of flowers):\n{target_names}")

# Features of dataset (more columns)
print(f"Feature names (description of each features):\n{iris_dataset['feature_names']}")

data = iris_dataset['data']
print(f"Type of data: {type(data)}")   # numpy.ndarray
print(f"Shape of data: {data.shape}")  

print(f"{data.shape[0]} instances of flowers each having {data.shape[1]} feature printed above")

print(f"First 5 rows (data pointes/samples) of data:\n{data[:5]}")

target = iris_dataset['target']
print(f"Type of target: {type(target)}")
print(f"Shape of target: {target.shape}")  # Only a single column

print(f"Target (Species code):\n{target}")
print("\n0: setosa, 1: versicolor, 2: virginica")

from sklearn.model_selection import train_test_split
# I've stored the values of dictionary keys in variables data and target
X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state = 0
)


print(f"X train shape: {X_train.shape}")  # 4 attributes
print(f"X test shape: {X_test.shape}")    # 4 attributes
print(f"y train shape: {y_train.shape}")  # 1 label
print(f"y test shape: {y_test.shape}")    # 1 label

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

# Model will be trained by this X_train and y_train training data

# Just predicting for one flower which have measurements: [5, 2.9, 1, 0.2]
# We've put this is 2D array because scikit-learn expects 2D array for the data.

X_new = np.array([[5, 2.9, 1, 0.2]])
print(f"Shape of X_new: {X_new.shape}")

prediction = knn.predict(X_new)

print(f"Prediction: {prediction}")
print(f"Predicted target (species) name: {target_names[prediction]}")

print(f"\nSo, according to our model, this new X_new flower is of type {target_names[prediction]}")

# Let's test for X_test data

y_pred = knn.predict(X_test)
print(f"Test set predictions:\n{y_pred}")

score = np.mean(y_pred == y_test)
print(f"Test set score: {score}")

print(f"Accuracy in percentage: {round(score * 100, 2)} %")