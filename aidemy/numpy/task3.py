import numpy as np

arr1 = [2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]

# unique()関数を用いて重複をなくした配列を変数new_arr1に代入してください
new_arr1 = np.unique(arr1)
print(new_arr1)

# 変数new_arr1と変数arr2の和集合を出力してください
print(np.union1d(new_arr1, arr2))

# 変数new_arr1と変数arr2の積集合を出力してください
print(np.intersect1d(new_arr1, arr2))

# 変数new_arr1から変数arr2を引いた差集合を出力してください
print(np.setdiff1d(new_arr1, arr2))