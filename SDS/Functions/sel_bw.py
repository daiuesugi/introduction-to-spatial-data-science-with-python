import numpy as np
from scipy.spatial import distance  # 距離行列の作成に必要なモジュール
from .commons import *

class Sel_BW:
  '''
  入力は被説明変数y, 説明変数X, 緯度経度coords
  '''

  # コンストラクタ
  def __init__(self, y, X, coords):
    self.y = np.matrix(y) # yの格納
    self.X = np.matrix(X) # Xの格納
    self.N = X.shape[0] # データ数
    self.K = X.shape[1] # 係数パラメータ数(定数項を含む)
    
    # 距離行列関連のリストの作成
    # 距離行列の作成(ユークリッド距離)
    dist_mat = distance.cdist(coords, coords, metric=distance.euclidean)
    # 昇順でdist_listを並べたリスト
    self.sorted_dist_list = np.sort(dist_mat, axis=1)
    # 昇順でdist_listを並べたインデックスのリスト
    sorted_dist_list_idx = np.argsort(dist_mat, axis=1)
    # 地点の順番をもとに戻すインデックスのリスト
    self.reverse_dist_list_idx = np.argsort(sorted_dist_list_idx, axis=1)


  # AICcを計算する関数
  def calc_aicc(self, bw_idx): # AICcを求める
    y = self.y
    X = self.X
    N = self.N # データ数
    # 重みリストの作成
    w_list = get_weight_list(N, self.sorted_dist_list, 
                             self.reverse_dist_list_idx, bw_idx)
    # 空のハット行列の作成
    S = np.zeros(shape=(N, N))

    for i in range(N):
      # 重み行列の作成(定義したget_weight_matrix()を使用)
      W_i = get_weight_matrix(w_list[i])
      # 地点iのハット行列の一部(R_i)を求める
      S[i, :] = np.matrix(X[i, :]) @ np.linalg.inv(X.T @ W_i @ X) @ X.T @ W_i

    # yの理論値
    y_hat = get_y_hat(y, S)
    
    # 残差
    resid = get_resid(y, y_hat)

    # 残差平方和
    rss = get_rss(resid)

    # 対数尤度
    mll = get_mll(y, y_hat, rss, N)
    
    # 有効パラメータ数
    v1 = get_v1(S)

    # AICc パラメータ数は(v1+1)
    aicc = get_aicc(mll, N, v1+1)

    return aicc


  # 黄金分割探索法
  def search(self):  # バンド幅の最小値bw_min・最大値bw_max
    y = self.y
    X = self.X
    
    # 黄金分割比
    gr = (np.sqrt(5) - 1) / 2  # おおよそ0.61803…

    # バンド幅の最大最小の初期値
    init_lb = 1  # バンド幅の最小値
    init_ub = self.N  # バンド幅の最大値

    # バンド幅の下限と上限の設定
    lb = init_lb
    ub = init_ub
    diff = 1e+5  # diffはy1とy2の差の絶対値(初期値は1e+5としている)

    while diff > 1e-5:  # 閾値は1e-5
      # 黄金分割費に基づいてバンド幅x1とx2 (x2 < x1) を決める
      # 整数のインデックスにするため、np.round()で四捨五入
      x1 = np.round(lb + gr * (ub - lb)).astype('int64')
      x2 = np.round(ub - gr * (ub - lb)).astype('int64')

      # y1(x1に対応するAICc)
      y1 = self.calc_aicc(x1)
      # y2(x2に対応するAICc)
      y2 = self.calc_aicc(x2)

      diff = np.abs(y1 - y2)  # diffはy1とy2の差の絶対値

      if y1 > y2:  # 走査する範囲の上限をx1に狭める
        lb = lb
        ub = x1
      else:  # 走査する範囲の下限をx2に狭める
        lb = x2
        ub = ub

    return(x1 + 1)