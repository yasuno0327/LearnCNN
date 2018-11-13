import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 今回は全データのうち、学習には300、テストには100個のデータを使用します
X_train = X_train[:300]
X_test = X_test[:100]
y_train = y_train[:300]
y_test = y_test[:100]

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
plt.suptitle('元画像', fontsize=12)
plt.show()

# ジェネレーターの生成
datagen = ImageDataGenerator(zca_whitening=True, featurewise_center=True)

# 白色化
datagen.fit(X_train[:10])
g = datagen.flow(X_train[:10], y_train[:10], shuffle=False)
X_batch, y_batch = g.next()

# 生成した画像を見やすくしています
X_batch *= 127.0 / max(abs(X_batch.min()), abs(X_batch.max()))
X_batch += 127
X_batch = X_batch.astype('uint8')

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_batch[i])
plt.suptitle('白色化結果', fontsize=12)
plt.show()