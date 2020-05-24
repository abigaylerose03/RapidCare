from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.linear_model as skl_lm # logistic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#machine-learning
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.metrics import confusion_matrix

sns.set(style="dark", color_codes=True, font_scale=1.3)

import pickle 

# loads the dataset with two classes: malignant (cancerous), benign (non-cancerous) tumors
breast_cancer_data = load_breast_cancer() 
breast_cancer_data.keys()

# load data into dataframe which contains rows x columns using Pandas
bc_dataset = pd.DataFrame(breast_cancer_data.data)
print(bc_dataset.shape) # (569, 30) which means 569 rows and 30 attributes
bc_dataset.columns = breast_cancer_data.feature_names # set columns of dataframe =  features like radius, texture, perimeter, etc
print(breast_cancer_data.feature_names) # the features: radius, texture perimester, smoothness, concavity, etc
print(breast_cancer_data.DESCR) # the description of the breast cancer wisconsin dataset\
print(bc_dataset.head())
print(bc_dataset.info())
print(breast_cancer_data.target_names)


# data cleaning
cols = ['worst radius', 
       	 'worst texture', 
        'worst perimeter', 
        'worst area', 
        'worst smoothness', 
        'worst compactness', 
        'worst concavity',
        'worst concave points', 
        'worst symmetry', 
        'worst fractal dimension']
bc_dataset = bc_dataset.drop(cols, axis=1)

cols = ['mean perimeter',
		 'perimeter error',
		 'mean area',
		 'area error']
bc_dataset = bc_dataset.drop(cols, axis=1)

cols = ['mean concavity',
        'concavity error', 
        'mean concave points', 
        'concave points error']

bc_dataset = bc_dataset.drop(cols, axis=1)
print(bc_dataset.columns)

X = bc_dataset # the data
y = breast_cancer_data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.75, random_state=45)

logisticRegressor = LogisticRegression()
logisticRegressor.fit(X_train, Y_train)

pickle.dump(logisticRegressor, open('model.pkl', 'wb')) # wb means written in binary just a reminder lol

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.23, 0.23, 0.235, 0.25, 0.235, -0.32, 0,23, -0.32, 10, 11, 12]])) # 0 means bengign and 1 means malignant

predictions = logisticRegressor.predict(X_test)
score = logisticRegressor.score(X_test, Y_test)
print(score)

cm = confusion_matrix(Y_test, predictions)

confusion_df = pd.DataFrame(confusion_matrix(Y_test,predictions),
             columns=["Predicted Class " + str(breast_cancer_data.target_names) for breast_cancer_data.target_names in [0,1]],
             index = ["Class " + str(breast_cancer_data.target_names) for breast_cancer_data.target_names in [0,1]])
# print(confusion_df)

# 4, 300, 500 test

# training_data, validation_data, training_labels, validation_labels = train_test_split(
#   breast_cancer_data.data,
#   breast_cancer_data.target,
#   test_size=0.2,
#   random_state=100
# )

# accuracies = []
# for k in range(1, 101):ß
#   classifier = KNeighborsClassifier(n_neighbors = k)
#   classifier.fit(training_data, training_labels)
#   accuracies.append(classifier.score(validation_data, validation_labels))

# k_list = range(1,101)
# plt.plot(k_list, accuracies)
# plt.title("Breast Cancer Classifier Accuracy")
# plt.xlabel("k")
# plt.ylabel("Validation Accuracy")
# plt.show()


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
