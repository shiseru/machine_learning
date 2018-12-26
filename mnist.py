import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./input/mnist.csv.zip")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1000)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
