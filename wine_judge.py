%matplotlib inline

#Classify Wine into tasty and not tasty wine from 11 different data categories

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./input/wine.csv')

df.head(2)

df1 = df[df['評価'] == 1]
df0  = df[df['評価'] == 0]

plt.scatter(df1['酢酸'], df1['クエン酸'], color='red')
plt.scatter(df0['酢酸'], df0['クエン酸'], color='blue')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


from sklearn.model_selection import train_test_split
# classify data into training and assessment data set
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state=0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000)
lr.fit(X_train, y_train)

lr.score(X_test, y_test)

