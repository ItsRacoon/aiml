import numpy as np  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split   
from sklearn import metrics 

# Define column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] 

# Read dataset to pandas DataFrame
dataset = pd.read_csv("D:\\lab\\aiml\\p5.csv")


# Features and target
X = dataset.iloc[:, :-1]   
y = dataset.iloc[:, -1] 

print(X.head())

# Split the data into training and testing sets (10% test data)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)

# Create and train the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5).fit(Xtrain, ytrain)  

# Predict the labels for the test set
ypred = classifier.predict(Xtest) 

# Print the results in a formatted table
i = 0 
print("\n-------------------------------------------------------------------------") 
print('{:<25} {:<25} {:<25}'.format('Original Label', 'Predicted Label', 'Correct/Wrong')) 
print("-------------------------------------------------------------------------") 

for label in ytest:
    print('{:<25} {:<25}'.format(label, ypred[i]), end="")
    if label == ypred[i]:
        print(' {:<25}'.format('Correct'))
    else:
        print(' {:<25}'.format('Wrong'))
    i += 1

print("-------------------------------------------------------------------------") 

# Confusion Matrix
print("\nConfusion Matrix:\n", metrics.confusion_matrix(ytest, ypred))   
print("-------------------------------------------------------------------------") 

# Classification Report
print("\nClassification Report:\n", metrics.classification_report(ytest, ypred))  
print("-------------------------------------------------------------------------") 

# Accuracy Score
print('Accuracy of the classifier is %0.2f' % metrics.accuracy_score(ytest, ypred)) 
print("-------------------------------------------------------------------------") 
