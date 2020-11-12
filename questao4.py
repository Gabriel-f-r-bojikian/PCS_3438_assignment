import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import KFold, cross_validate
from numpy import average

#Importing data
filepath = 'reg02.csv'

data = pd.read_csv(filepath)

model = DecisionTreeRegressor(criterion = 'mse')

#Fit the model and evaluate
'''
Arguments:
  n_splits = 5 -> 5 splits in data
  shuffle = False -> Will not shuffle data for the splits and use it in order
  random_state = None -> Seed that would be used for shuffling the dataset
'''
k_fold = KFold(n_splits = 5, shuffle = False, random_state = None)
results = cross_validate(model, X=data.drop(['target'], axis = 1), y=data['target'], scoring = 'neg_median_absolute_error', cv=k_fold, return_train_score = True)

scores = results['test_score']

print("Validation results average: {}".format(average(-results['test_score'])))
print("Training results average: {}".format(average(-results['train_score']) ) )