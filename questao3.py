import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut, cross_validate
from numpy import average

data = pd.read_csv('reg01.csv')

model = Lasso(alpha = 1)

loo = LeaveOneOut()
results = cross_validate(model, X=data.drop(['target'], axis = 1), y=data['target'], scoring = 'neg_root_mean_squared_error', cv=loo, return_train_score = True)

print('Training set avg RMSE: {}'.format(-average(results['train_score'])))
print('Validation set avg RMSE: {}'.format(-average(results['test_score'])))
