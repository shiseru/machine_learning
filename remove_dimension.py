%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

# 画像データの読込
df = pd.read_csv('./input/mnist.csv.zip')

data = df.iloc[:, :-1].values
data.shape

plt.imshow(-data[0].reshape(28, 28), cmap='gray');

from sklearn.decomposition import PCA

N = 200 # only 200 images in folder
pca = PCA(N)
result = pca.fit(data)
trans_data = result.transform(data)

trans_data[:, -1].round(10)

plt.scatter(trans_data[:, 0], trans_data[:, 1])

def show_image(result, i):
    plt.imshow(result.components_[i].reshape(28, 28), cmap='gray')

show_image(result, 0)

show_image(result, 1)
show_image(result, 2)
show_image(result, 3)
show_image(result, 4)