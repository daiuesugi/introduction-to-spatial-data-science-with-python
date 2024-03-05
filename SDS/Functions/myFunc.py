import numpy as np

# 掛け算の関数
def prod(x1, x2):
  y = x1 * x2
  return y

# リストをnp.array型に変換する関数
# 入力x(引数)はリスト
def list_to_array(x):
  y = np.array(x)
  return y