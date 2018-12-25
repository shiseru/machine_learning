import pandas as pd
df = pd.read_csv('input/data.csv')


# example that Linear Regression cannot be used on Linear non-separable question (XOR graph)

X = df[['x1', 'x2']]
y = df['y']

from sklearn.model_selection import train_test_split
(X_train, X_test,
 y_train, y_test) = train_test_split(
 X, y, test_size=0.3, random_state=0,
)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000)

lr.fit(X_train, y_train)

lr.score(X_test, y_test)

%matplotlib inline
import matplotlib.pyplot as plt
X_train_0 = X_train[y_train == 0]
X_train_1 = X_train[y_train == 1]

ax = plt.figure().subplots()
X_train_0.plot.scatter('x1', 'x2', color='red', ax=ax)
X_train_1.plot.scatter('x1', 'x2', color='blue', ax=ax);


def judge(x1, x2):
    if(x1 > 0) ^ (x2 > 0):
        return 1
    else:
        return 0

answers = []

for row in X_test.itertuplus():
    answers.append(judge(row.x1, row.x2))

(answers == y_test).sum() / len(y_test)



# from 2nd dimension XOR plt, change it to 3rd dimension to solve

%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('input/data.csv')

X = df[['x1', 'x2']]
y = df['y']

x_0 = X[y == 0]
x_1 = X[y == 1]

ax = plt.figure().subplots()


%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('input/data.csv')

X = df[['x1', 'x2']]
y = df['y']

X_0 = X[y == 0]
X_1 = X[y == 1]

ax = plt.figure().subplots()
X_0.plot.scatter('x1', 'x2', color = 'red', ax=ax)
X_1.plot.scatter('x1','x2', color = 'blue', ax=ax);

import numpy as np

new_column = X.x1 * X.x2
extend_X = np.c_[X, new_column]
extend_X.shape

extend_X[:2]

from sklearn.model_selection import train_test_split
(X_train, X_test,
 y_train, y_test) = train_test_split(
    extend_X, y, test_size=0.3, random_state=0
)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100)

lr.fit(X_train, y_train)

lr.score(X_test, y_test)






