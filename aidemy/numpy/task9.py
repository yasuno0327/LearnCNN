import numpy as np

# arrを定義します
arr = np.arange(9).reshape(3, 3)

# 変数arrとarrの行列積を出力してください
print(np.dot(arr, arr))

# vecを定義します
vec = arr.reshape(9)

# 変数vecのノルムを出力してください
print(np.linalg.norm(arr))