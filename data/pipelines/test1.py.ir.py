

def __phi__(phi_0, phi_1):
    if phi_0:
        return phi_0
    return phi_1

def set_field_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def set_index_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def global_wrapper(x):
    return x
import pandas as pd
import numpy as np
_var0 = 'data.csv'
df = pd.read_csv(_var0)
l = len(df)
_var1 = 'Survived'
y = df[_var1]
_var2 = 'Survived'
_var3 = 1
df_0 = df.drop(_var2, axis=_var3)
from sklearn.model_selection import train_test_split
_var8 = 0.33
_var9 = 42
(_var4, _var5, _var6, _var7) = train_test_split(df_0, y, test_size=_var8, random_state=_var9)
X_train = _var4
X_test = _var5
y_train = _var6
y_test = _var7
_var10 = 'Fare'
_var11 = 'Fare'
_var12 = X_train[_var11]
_var13 = 'Fare'
_var14 = X_train[_var13]
_var15 = np.mean(_var14)
_var16 = False
_var17 = _var12.fillna(_var15, inplace=_var16)
X_train_0 = set_index_wrapper(X_train, _var10, _var17)
_var18 = 'Fare'
_var19 = 'Fare'
_var20 = X_test[_var19]
_var21 = 'Fare'
_var22 = X_test[_var21]
_var23 = np.mean(_var22)
_var24 = False
_var25 = _var20.fillna(_var23, inplace=_var24)
X_test_0 = set_index_wrapper(X_test, _var18, _var25)
from sklearn.linear_model import LogisticRegression
_var26 = 'l2'
_var27 = 'sag'
_var28 = 0
clf = LogisticRegression(penalty=_var26, solver=_var27, random_state=_var28)
clf_0 = clf.fit(X_train_0, y_train)
clf_0.predict(X_train_0)
y_pred = clf_0.predict(X_test_0)
