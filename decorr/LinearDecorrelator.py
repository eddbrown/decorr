import pandas as pd
import numpy as np

class LinearDecorrelator:
    def __init__(self, correlated_feature, data_frame):
        self.cf = correlated_feature
        self.df = data_frame

    def decorrelate(self):
        for col in self.df:
            coeff = np.corrcoef(self.cf, self.df[col])[1,0]
            [gradient, intercept] = np.polyfit(self.cf, self.df[col],1)
            new_col = self.df[col].values
            new_col = new_col - intercept
            new_col = new_col - (self.cf * gradient)
            self.df[col] = new_col
            new_coeff = np.corrcoef(self.cf, new_col)[1,0]
            print(new_coeff)
        return self.df
