import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1250, n_features=4, n_informative=2, n_redundant=2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを書いてください。
# 乱数生成器の構築
#ここに答えを書いてください
random_state = np.random.RandomState()

# モデルの構築
#ここに答えを書いてください
model = SVC(random_state=random_state)

# モデルの学習
#ここに答えを書いてください
model.fit(train_X, train_y)

# テストデータに対する正解率を出力
print(model.score(test_X, test_y))