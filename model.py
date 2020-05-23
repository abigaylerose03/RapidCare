from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dataset import load_breast_cancer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the dataset with two classes: malignant (cancerous), benign (non-cancerous) tumors
breast_cancer_data = load_breast_cancer() 
breast_cancer_data.keys()
breast_cancer_data.shape
print(breast_cancer_data.feature_names)
print(breast_cancer_data.DESR)

training_data, validation_data, training_labels, validation_labels = train_test_split(
  breast_cancer_data.data,
  breast_cancer_data.target,
  test_size=0.2,
  random_state=100
)

accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

k_list = range(1,101)
plt.plot(k_list, accuracies)
plt.title("Breast Cancer Classifier Accuracy")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()

# overfitting means relying too much on training data
#underfitting not relying enough on the training data
# 80% data in training set 20% validation set

# print(breast_cancer_data.feature_names)

# print(len(training_data))
# print(len(training_labels))

# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)
# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)
