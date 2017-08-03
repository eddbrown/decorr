import pandas as pd
import numpy as np

class LinearDecorrelator:
    def __init__(self):
        self.feature_dict = {}
        self.v = 'Not set'
        self.features = []

    def fit(self,v,features,df):
        self.features = features
        self.v = v
        self.feature_dict = {}
        for f in features:
            [gradient, intercept] = np.polyfit(df[v], df[f],1)

            self.feature_dict[f] = {
                'gradient': gradient,
                'intercept': intercept,
            }

    def transform(self,df):
        for f in self.features:
            if self.feature_dict.get(f, False) != False:
                gradient = self.feature_dict[f]['gradient']
                intercept = self.feature_dict[f]['intercept']
                # import code; code.interact(local=dict(globals(), **locals()))
                new_values = df.loc[:,f].values - intercept - (df.loc[:,self.v].values * gradient)
                df.loc[:,f] = new_values
