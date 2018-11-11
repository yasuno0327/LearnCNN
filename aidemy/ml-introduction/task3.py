# 混合行列
# 今回必要となるモジュールをインポートします
import numpy
from sklearn.metrics import confusion_matrix

# データを格納します。今回は0が陽性、1が陰性を示しています
y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

# 以下の行に変数confmatにy_trueとy_predの混合行列を格納してください
confmat = confusion_matrix(y_true, y_pred)

# 結果を出力します。
print (confmat)