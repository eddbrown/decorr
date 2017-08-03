# Decorr Package

This Package aims to help data scientist using python to easily decorrelate their data set from a variable.

# LinearDecorrelator

The first class for this package is the LinearDecorrelator class.
The functionality follows the scikit learn format of fit and transform

```python
ld = LinearDecorrelator()
ld.fit(controlling_variable,dataframe)

new_dataframe = ld.transfrom(controlling_variable,dataframe)
```
