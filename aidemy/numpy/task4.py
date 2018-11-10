import numpy as np
from numpy.random import randint
from numpy.random import rand
# randint関数をnp.randomとつけなくてもいいようにimportしてください


# 変数arr1に、各要素が0 ~ 10まで整数の行列(5 × 2)を代入してください
arr1 = randint(0, 11, (5, 2))
print(arr1)

# 変数arr2に0~1までの一様乱数を三つ代入してください
arr2 = rand(3)
print(arr2)