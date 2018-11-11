from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets

# データを取得してください
iris = datasets.load_iris()
# irisの0列目と2列目を格納してください
X = iris.data[:, [0, 2]]
# irisのクラスラベルを格納してください
y = iris.target

# データの色付け、プロット
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="cool"), alpha=0.7)
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.grid(True)
plt.show()
