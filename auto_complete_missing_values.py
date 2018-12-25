import pandas as pd
df = pd.read_csv('./input/data1.csv')

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="mean")

imp.fit(df[["身長(cm)"]])

values = imp.transform(df[["身長(cm)"]])

imp = Imputer(missing_values="NaN", strategy="median")
values = imp.fit_transform(df[["体重(kg)"]])
df[["体重(kg)"]] = values

imp = Imputer(missing_values="NaN", strategy="most_frequent")

values = imp.fit_transform(df[["視力"]])

df[["視力"]] = values