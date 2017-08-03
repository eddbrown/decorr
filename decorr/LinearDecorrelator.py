import pandas as pd
import numpy as np

class LinearDecorrelator:
    def __init__(self):
        self.feature_dict = {}

    def fit(self,cf,df):
        self.feature_dict = {}
        for col in df:
            [gradient, intercept] = np.polyfit(cf, df[col],1)

            self.feature_dict[col] = {
                'gradient': gradient,
                'intercept': intercept,
            }

    def transform(self,cf,df):
        for col in df:
            gradient = self.feature_dict[col]['gradient']
            intercept = self.feature_dict[col]['intercept']
            df[col] = df[col] - intercept
            df[col] = df[col] - (cf * gradient)

            new_coeff = np.corrcoef(cf, df[col])[1,0]
            print(new_coeff)

        return df
