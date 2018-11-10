import numpy as np

arr = np.array([2, 3, 4, 5, 6, 7])

# 変数arrの各要素が2で割り切れるかどうかを示す真偽値の配列を出力してください
print(arr % 2 == 0)

# 変数arr各要素のうち2で割り切れる要素の配列を出力してください
print(arr[arr % 2 == 0])