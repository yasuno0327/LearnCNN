import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ごくシンプルな畳み込み層を定義しています。
class Conv:
    # シンプルな例を考えるため、Wは3x3で固定し、後のセッションで扱うstridesやpaddingは考えません。
    def __init__(self, W):
        self.W = W
    def f_prop(self, X):
        out = np.zeros((X.shape[0]-2, X.shape[1]-2))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                x = X[i:i+3, j:j+3]
                out[i,j] = np.dot(self.W.flatten(), x.flatten())
        return out

# ごくシンプルなプーリング層を定義しています。
class Pool:
    # シンプルな例を考えるため、後のセッションで扱うstridesやpaddingは考えません。
    def __init__(self, l):
        self.l = l
    def f_prop(self, X):
        l = self.l
        out = np.zeros((X.shape[0]//self.l, X.shape[1]//self.l))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                # 下の下線部を埋めて、コメントアウトをはずしてください。
                out[i,j] = np.max(X[i*l:(i+1)*l, j*l:(j+1)*l])
        return out

local_filename, headers = urllib.request.urlretrieve('https://aidemyexcontentsdata.blob.core.windows.net/data/5100_cnn/circle.npy')
X = np.load(local_filename)

plt.imshow(X)
plt.title("元画像", fontsize=12)
plt.show()

# カーネル
W1 = np.array([[0,1,0],
               [0,1,0],
               [0,1,0]])
W2 = np.array([[0,0,0],
               [1,1,1],
               [0,0,0]])
W3 = np.array([[1,0,0],
               [0,1,0],
               [0,0,1]])
W4 = np.array([[0,0,1],
               [0,1,0],
               [1,0,0]])

# 畳み込み
conv1 = Conv(W1); C1 = conv1.f_prop(X)
conv2 = Conv(W2); C2 = conv2.f_prop(X)
conv3 = Conv(W3); C3 = conv3.f_prop(X)
conv4 = Conv(W4); C4 = conv4.f_prop(X)

plt.subplot(1,4,1); plt.imshow(C1)
plt.subplot(1,4,2); plt.imshow(C2)
plt.subplot(1,4,3); plt.imshow(C3)
plt.subplot(1,4,4); plt.imshow(C4)
plt.suptitle("畳み込み結果", fontsize=12)
plt.show()

# プーリング
pool = Pool(2)
P1 = pool.f_prop(C1)
P2 = pool.f_prop(C2)
P3 = pool.f_prop(C3)
P4 = pool.f_prop(C4)

plt.subplot(1,4,1); plt.imshow(P1)
plt.subplot(1,4,2); plt.imshow(P2)
plt.subplot(1,4,3); plt.imshow(P3)
plt.subplot(1,4,4); plt.imshow(P4)
plt.suptitle("プーリング結果", fontsize=12)
plt.show()