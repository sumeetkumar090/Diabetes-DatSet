# Classification Random Forest template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,8].values


#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 2:6])
X[:, 2:6] = imputer.transform(X[:, 2:6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) 

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()


#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{ 'n_neighbors':[2,3,4,5,6,7,8,9,10], 'algorithm':['ball_tree'], 'p':[1,2], 'metric':['minkowski']},
             { 'n_neighbors':[2,3,4,5,6,7,8,9,10], 'algorithm':['kd_tree'], 'p':[1,2], 'metric':['minkowski']},
             { 'n_neighbors':[2,3,4,5,6,7,8,9,10], 'algorithm':['brute'], 'p':[1,2], 'metric':['minkowski']},
                         
              ]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
Std_dev = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)



