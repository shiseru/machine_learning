import pandas as pd

df = pd.read_csv('input/reviews.csv')


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(df['text'])
y = df.iloc[:, -1].values()

from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1000)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

