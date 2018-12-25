import pandas as pd
df = pd.read_csv('./input/data2.csv')

size_mapping = {
    'XL' : 4, 'L' : 3, 'M' : 2, 'S' : 1
}

values = df['size'].map(size_mapping)

df['size'] = values


import pandas as pd
df = pd.read_csv('./input/data2.csv')

classlabels = list(set(df["classlabel"]))

class_mapping = {}

for number, label in enumerate(sorted(classlabels)):
    class_mapping[label] = number

classlabels_data = df['classlabel'].map(class_mapping)
df['calsslabel'] = classlabels_data

reverse_class_mapping = {values: key for key, value in class_mapping.item()}

df['classlabel'] = df['classlabel'].map(reverse_class_mapping)
df


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

classlabels_data = encoder.fit_transform(df['calsslabel'])
df['calsslabel'] = classlabels_data

df['classlabel'] = encoder.classes_[classlabels_data]
df

import pandas as pd
df = pd.read_csv('./input/data2.csv')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

values = encoder.fit_transform(df['color'])

values.reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
result = encoder.fit_transform(values.reshape(-1, 1))

colors_df = pd.DataFrame(result.toarray(),
                         columns=sorted(list(set(df['color']))))


import pandas as pd
df = pd.read_csv('./input/data2.csv')

colors_df = pd.get_dummies(df['color'])

df = pd.merge(df, colors_df, left_index=True, right_index=True, how='outer')




