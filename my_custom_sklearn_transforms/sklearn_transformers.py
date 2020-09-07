from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class UnderSampling(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target
    def fit(self,X, y):
        return self

    def transform(self,X):
        under= RandomUnderSampler(sampling_strategy=0.2)
        dataX = [X.columns]
        return under.fit_sample(dataX,self.target)

