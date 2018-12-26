%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('input/data14.csv')

X = df[['x0', 'x1']].values
y = df['y'].values

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

tree = DecisionTreeClassifier(max_depth = 4)

result = cross_validate(
    estimator=tree, X=X, y=y, cv=3, return_train_score=True
)

#check the correctness of decision tree by cross validation
def check_tree(X, y, max_depth, cv):
    import numpy as np
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    result = cross_validate(
        estimator=tree, X=X, y=y, cv=cv, return_train_score=True
    )
    return tree, np.mean(result['train_score']), np.mean(result['test_score'])

max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
score_trains = []
score_tests = []

for max_depth in max_depths:
    _, score_train, score_test = check_tree(X, y, max_depth, cv=3)
    print('max_depth', max_depth, 'train', score_train, 'test', score_test)
    score_trains.append(score_train)
    score_tests.append(score_test)

plt.xlabel('max_depth')
plt.plot(max_depths, score_trains, label='train')
plt.plot(max_depths, score_tests, label='test')
plt.legend();