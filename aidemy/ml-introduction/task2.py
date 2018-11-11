# コードの実行に必要なモジュールを読み込みます。
from sklearn import svm, datasets, cross_validation

# 「IRIS」というデータセットを読み込みます。
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 機械学習アルゴリズムSVMを使用
svc = svm.SVC(C=1, kernel="rbf", gamma=0.001)

# 交差検証法を用いてスコアを求めます。
# 内部では、X、yがそれぞれ「X_train, X_test, y_train, y_test」の様に分割され処理されます。
scores = cross_validation.cross_val_score(svc, X, y, cv=5)

# トレーニングデータとテストデータのサイズを確認します。
print (scores)
print ("平均スコア :", scores.mean())