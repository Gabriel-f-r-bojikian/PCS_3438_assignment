import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_validate
from numpy import average

#Importing data
filepath = 'class02.csv'

data = pd.read_csv(filepath)

'''
Explaining the arguments of the classifier:
  n_neighbors = 5 -> Will use the 5 closest neighbors for classification
  weights = 'uniform' -> All votes have the same value
  metric = 'minkowski' and p = 2 -> Make the model use euclidean distance 
'''
model = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', p = 2, metric = 'minkowski')

#Fit the model and evaluate
'''
Arguments:
  n_splits = 5 -> 5 splits in data
  shuffle = False -> Will not shuffle data for the splits and use it in order
  random_state = None -> Seed that would be used for shuffling the dataset
'''
k_fold = KFold(n_splits = 5, shuffle = False, random_state = None)
results = cross_validate(model, X=data.drop(['target'], axis = 1), y=data['target'], scoring = 'accuracy', cv=k_fold, return_train_score = True)

scores = results['test_score']

print('\n')
print("Validation results: {}".format( average(results['test_score'])))
print("Train results {}".format( average(results['train_score'])))