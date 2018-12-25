%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('input/data14.csv')

plt.scatter(df[df.y == 0].x0, df[df.y == 0].x1)
plt.scatter(df[df.y == 1].x0, df[df.y == 1].x1);

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


from sklearn.model_selection import train_test_split
(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, y, test_size=0.3, random_state=0
)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

tree.score(X_test, y_test)
# 0.72 for test, 1.0 for practice - overfitting

from plot_tree import plot_tree
plot_tree(tree, X, y)

# set max depth to prevent overfitting
from sklearn.tree import DecisionTreeClassifier
tree_depth3 = DecisionTreeClassifier(max_depth=3)
tree_depth3.fit(X_train, y_train)

tree_depth3.score(X_test, y_test)

plot_tree(tree_depth3, X, y)

