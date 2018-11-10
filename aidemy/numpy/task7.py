import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 12], [15, 20, 22]])
arr2 = np.arange(25).reshape(5, 5)

# arrの行の合計値を求め、問題文の1次元配列を返してください
print(arr.sum(axis=1))

# 変数arrの行の順番を変更したものを出力してください
print(arr2[[1, 3, 0]])

# 変数arrを転置させてください
print(arr2.T)