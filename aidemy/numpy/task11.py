import numpy as np


# 0から14の整数値をもつ3×5のndarray配列xを生成
x = np.arange(15).reshape(3, 5)

# 0から4の整数値をもつ1×5のndarray配列yを生成
y = np.array([np.arange(5)])

# xのn番目の列のすべての要素に対してnだけ引いてください
z = x - y

# xを出力
print(z)