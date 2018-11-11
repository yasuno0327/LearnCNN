# 適合率、再現率、F1
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

# データを格納します。今回は0が陰性、1が陽性を示しています
y_true = [1,1,1,0,0,0]
y_pred = [0,1,1,0,0,0]

# 適合率と再現率をあらかじめ計算します
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# 以下の行にF1スコアの定義式を書いてください
f1_score = 2 * precision * recall / (precision + recall)

print("F1: %.3f" % f1_score)