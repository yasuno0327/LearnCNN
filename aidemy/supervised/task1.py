# モジュールのimport
from sklearn.datasets import make_classification
# プロット用モジュール
import matplotlib.pyplot as plt
import matplotlib

# データX, ラベルyを生成
X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)


# データの色付け、プロット
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)
plt.grid(True)
plt.show()
