from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder(sparse=False,n_values=2)
ohe.fit(np.array([1,0]).reshape(-1,1))
results = ohe.transform(np.array([1,1,1,0,1,0,0]).reshape(-1,1))
print(results.dtype)

