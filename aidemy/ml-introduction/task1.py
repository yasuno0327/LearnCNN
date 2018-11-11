# コードの実行に必要なモジュールを読み込みます。
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 「IRIS」というデータセットを読み込みます。
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 「X_train, X_test, y_train, y_test」にデータを格納します。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# トレーニングデータとテストデータのサイズを確認します。
print ("X_train :", X_train.shape)
print ("y_train :", y_train.shape)
print ("X_test :", X_test.shape)
print ("y_test :", y_test.shape)