

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
import numpy as np
import pandas as pd
_var0 = '../input/train.csv'
train = pd.read_csv(_var0)
_var1 = '../input/test.csv'
test = pd.read_csv(_var1)
train.head()
test.head()
_var2 = [train, test]
_var3 = False
data = pd.concat(_var2, sort=_var3)
data.head()
_var4 = len(train)
_var5 = len(test)
_var6 = len(data)
print(_var4, _var5, _var6)
_var7 = data.isnull()
_var7.sum()
_var8 = 'Sex'
_var9 = data[_var8]
_var10 = ['male', 'female']
_var11 = [0, 1]
_var12 = True
_var9.replace(_var10, _var11, inplace=_var12)
_var13 = 'Embarked'
_var14 = data[_var13]
_var15 = 'S'
_var16 = True
_var14.fillna(_var15, inplace=_var16)
_var17 = 'Embarked'
_var18 = 'Embarked'
_var19 = data[_var18]
_var20 = {'S': 0, 'C': 1, 'Q': 2}
_var21 = _var19.map(_var20)
_var22 = _var21.astype(int)
data_0 = set_index_wrapper(data, _var17, _var22)
_var23 = 'Fare'
_var24 = data_0[_var23]
_var25 = 'Fare'
_var26 = data_0[_var25]
_var27 = np.mean(_var26)
_var28 = True
_var24.fillna(_var27, inplace=_var28)
_var29 = 'Age'
_var30 = data_0[_var29]
age_avg = _var30.mean()
_var31 = 'Age'
_var32 = data_0[_var31]
age_std = _var32.std()
_var33 = 'Age'
_var34 = data_0[_var33]
_var35 = np.random
_var36 = (age_avg - age_std)
_var37 = (age_avg + age_std)
_var38 = _var35.randint(_var36, _var37)
_var39 = True
_var34.fillna(_var38, inplace=_var39)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
_var40 = 1
_var41 = True
data_0.drop(delete_columns, axis=_var40, inplace=_var41)
_var42 = len(train)
train_0 = data_0[:_var42]
_var43 = len(train_0)
test_0 = data_0[_var43:]
_var44 = 'Survived'
y_train = train_0[_var44]
_var45 = 'Survived'
_var46 = 1
X_train = train_0.drop(_var45, axis=_var46)
_var47 = 'Survived'
_var48 = 1
X_test = test_0.drop(_var47, axis=_var48)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
_var49 = 'l2'
_var50 = 'sag'
_var51 = 0
clf = LogisticRegression(penalty=_var49, solver=_var50, random_state=_var51)
clf_0 = clf.fit(X_train, y_train)
y_pred = clf_0.predict(X_test)
