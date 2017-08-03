import pandas as pd
import numpy as np

class PolynomialDecorrelator:
    def __init__(self,degree=2):
        self.degree = 2
        self.feature_dict = {}
        self.v = 'Not set'
        self.features = []

    def fit(self,v,features,df):
        self.features = features
        self.v = v
        self.feature_dict = {}
        for f in features:
            params = np.polyfit(df[v], df[f],self.degree)

            self.feature_dict[f] = {
                'params': params,
            }

    def transform(self,df):
        for f in self.features:
            if self.feature_dict.get(f, False) != False:
                params = self.feature_dict[f]['params']

                for i,param in enumerate(params, start=0):
                    degree = self.degree - i

                    df.loc[:,f] = df.loc[:,f].values - param * (df.loc[:,self.v].values ** degree)
