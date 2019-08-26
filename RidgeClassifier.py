import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 8].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 2:6])
X[:, 2:6] = imputer.transform(X[:, 2:6])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import RidgeClassifier
classifier = RidgeClassifier(alpha=.1, fit_intercept=True,random_state=0,solver='auto',tol=0.010)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{ 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['svd']   },
              { 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['cholesky']   },
              { 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['sparse_cg']   },
              { 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['lsqr']   },
              { 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['sag']   },
              { 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5], 'tol':[0.01,0.02,0.03,0.04,0.05], 'solver':['saga']   },
              ]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

