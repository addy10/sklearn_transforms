from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self
    
    def transform(self, X,y):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        under = RandomUnderSampler(sampling_strategy=0.5)
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return sc.fit_transform(data)
    
class OverSampling(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self
    
    def transform(self, X,y):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        over = RandomOverSampler(sampling_strategy=0.5)
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return sc.fit_transform(data)
