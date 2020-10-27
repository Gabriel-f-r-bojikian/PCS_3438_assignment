import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

filepath = "class01.csv"

data = pd.read_csv(filepath)

#Separação dos dados: Validação cruzada Holdout
X_treino = data.iloc[:350].drop(labels = 'target', axis = 1)
y_treino = data.iloc[:350, -1]

X_validacao = data.iloc[351:].drop(labels = 'target', axis = 1)
y_validacao = data.iloc[351:, -1]

#Criação e treino do modelo
Naive_Bayes_Regressor = GaussianNB()
Naive_Bayes_Regressor.fit(X_treino, y_treino)

#Validação
acuracia_treino = accuracy_score(y_true = y_treino , y_pred = Naive_Bayes_Regressor.predict(X_treino))
acuracia_validacao = accuracy_score(y_true = y_validacao , y_pred = Naive_Bayes_Regressor.predict(X_validacao))

print('\nAcurácia treino [%]: {}\nAcurácia validação [%]: {}'.format(acuracia_treino, acuracia_validacao))