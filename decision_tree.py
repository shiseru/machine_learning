%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

# 1. データの読み込み
df = pd.read_csv('./input/data13.csv')
df.head(3)

plt.scatter(df[df['y'] == 0]['x0'], df[df['y'] == 0]['x1'])
plt.scatter(df[df['y'] == 1]['x0'], df[df['y'] == 1]['x1']);

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
## あとでメモ
(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, y, test_size=0.3, random_state=0
)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

tree.score(X_test, y_test)

from plot_tree import  plot_tree
plot_tree(tree, X, y)
