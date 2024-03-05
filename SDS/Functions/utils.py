import numpy as np

# obj_name : geopandasデータフレームのオブジェクト名
# y_data : geopandasデータフレームの列名をリストにする  ex) georgia['PctBach']
# X_data : geopandasデータフレームの列名をリストにする  ex) georgia[['PctFB', 'PctBlack', 'PctRural']]
# pos_data : geopandasデータフレームの列名をリストにする  ex) georgia[['X', 'Y']]
# std : y_data,X_dataを標準化するか否か(デフォルトはFalse)

def get_input_data(y_data, X_data, pos_data, std=False):
  # データ数
  N = len(y_data)

  # データフレームをnp.arrayに変換する
  X = X_data.values.astype(np.float64)

  # Xを標準化
  if std == True:
    X = standardize(X)

  # 定数項の列(すべて1)を作成する
  intercept = np.ones(N).reshape([-1, 1])
  # 2つの配列を水平方向に結合する
  X = np.hstack([intercept, X])

  # y(被説明変数)をN×1配列に変換
  y = y_data.values.reshape(-1, 1).astype(np.float64)

  # yを標準化
  if std == True:
    y = standardize(y)

  # 位置情報(緯度経度)データのリストを作成する
  u = pos_data.iloc[:, 0].astype(np.float64)
  v = pos_data.iloc[:, 1].astype(np.float64)
  coords = list(zip(u, v))

  return y, X, coords

def standardize(x):
  x = (x - x.mean(axis=0)) / x.std(axis=0)
  return x