# Remove missing, useless values to train well

import pandas as pd
df = pd.read_csv('./input/data1.csv')

# remove_row
df1 = df.dropna()

# remove_column
df2 = df.dropna(axis=1)

# Remove the row which the satisfied data is less then 4
df3 = df.dropna(thresh=4)

# Remove the row where 身長cm is Null
df4 = df.dropna(subset=["身長(cm)"])

