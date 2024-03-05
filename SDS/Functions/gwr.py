import numpy as np
from .ols import OLS
from scipy.spatial import distance  # 距離行列の作成に必要なモジュール
from .commons import *


class GWR(OLS):
  """
    GWR推定を行うクラス
  """
  def __init__(self, y, X, coords, bw):
    # メンバ変数
    super().__init__(y, X)  # 親クラスのOLSのメンバ変数も利用
    # 追加のメンバ変数
    self.bw = bw  # バンド幅
    self.bw_idx = bw - 1  # バンド幅(インデックス)
    self.v1 = None  # 有効パラメータ数
    self.alpha_c = None  # 修正α
    self.tval_c = None  # 修正t値
    self.local_R2 = None  # ローカル決定係数
    self.local_std = None  # パラメータ推定値の標準誤差
    self.local_t = None  # パラメータ推定値のt値
    
    # 距離行列関連のリストの作成
    # 距離行列の作成(ユークリッド距離)
    dist_mat = distance.cdist(coords, coords, metric=distance.euclidean)
    # 昇順でdist_listを並べたリスト
    self.sorted_dist_list = np.sort(dist_mat, axis=1)
    # 昇順でdist_listを並べたインデックスのリスト
    sorted_dist_list_idx = np.argsort(dist_mat, axis=1)
    # 地点の順番をもとに戻すインデックスのリスト
    self.reverse_dist_list_idx = np.argsort(sorted_dist_list_idx, axis=1)
    
    # 重みリストの作成
    self.w_list = get_weight_list(self.N, self.sorted_dist_list, 
                                  self.reverse_dist_list_idx, self.bw_idx)


  # 係数パラメータ(beta_hat)を求める関数
  def get_beta_hat(self):
    y = self.y
    X = self.X
    N = self.N
    w_list = self.w_list
        
    # パラメータ推定値の格納オブジェクトbeta_hatを作成
    beta_hat = np.zeros_like(X)

    for i in range(N):
      # 重み行列の作成(定義したget_weight_matrix()を使用)
      W_i = get_weight_matrix(w_list[i])
      # 地点iの係数パラメータを求める
      beta_i_hat = np.linalg.inv(X.T @ W_i @ X) @ X.T @ W_i @ y
      beta_hat[i, :] = beta_i_hat.flatten()

    return beta_hat


  # ハット行列(S)を求める関数
  def get_S(self):
    y = self.y
    X = self.X
    N = self.N
    w_list = self.w_list
    
    # 空のハット行列の作成
    S = np.zeros(shape=(N, N))

    for i in range(N):
      # 重み行列の作成(定義したget_weight_matrix()を使用)
      W_i = get_weight_matrix(w_list[i])
      # 地点iのハット行列の一部(R_i)を求める
      S[i, :] = np.matrix(X[i, :]) @ np.linalg.inv(X.T @ W_i @ X) @ X.T @ W_i
    
    return S
  
  
  # モデルを推定する関数
  def fit(self):
    self.beta_hat = self.get_beta_hat()
    self.S = self.get_S()
    self.v1 = get_v1(self.S)
    self.df = self.N - self.v1
    self.y_hat = get_y_hat(self.y, self.S)
    self.resid = get_resid(self.y, self.y_hat)
    self.rss = get_rss(self.resid)
    self.sigma2 = get_sigma2(self.rss, self.N, self.v1)
    self.sigma_hat = np.sqrt(self.sigma2)
    self.mll = get_mll(self.y, self.y_hat, self.rss, self.N)
    self.aic = get_aic(self.mll, self.v1+1)
    self.aicc = get_aicc(self.mll, self.N, self.v1+1)
    self.bic = get_bic(self.mll, self.N, self.v1+1)
    self.R2, self.adjR2 = get_R2(self.y, self.rss, self.N, self.v1+1)
    self.alpha_c = get_alpha_c(self.K, self.v1, xi=0.05)
    self.tval_c = get_tval_c(self.N, self.K, self.alpha_c)
    self.local_R2 = get_local_R2(self.y, self.y_hat, self.N, self.w_list)
    self.local_std, self.local_t = get_local_std_and_local_t(self.X, self.beta_hat, self.sigma2, self.N, self.w_list)


  # 結果の要約を表示する関数
  def summary(self):
    XNames = ["X" + str(k) for k in range(self.K)]  # 変数名のラベル
    beta_hat_mean = np.mean(self.beta_hat, axis=0)  # パラメータ推定値の平均
    beta_hat_std = np.std(self.beta_hat, axis=0)  # 標準誤差
    beta_hat_min = np.min(self.beta_hat, axis=0)  # 最小値
    beta_hat_median = np.median(self.beta_hat, axis=0)  # 中央値
    beta_hat_max = np.max(self.beta_hat, axis=0)  # 最大値
    
    summary = '{}\n'.format('Geographically Weighted Regression (GWR) Results')
    summary += '=' * 75 + '\n'
    summary += '{0:55}{1:>20}\n'.format('Spatial kernel:', 'Adaptive bisquare')
    summary += '{0:60}{1:15.0f}\n'.format('Bandwidth used:', self.bw)
    summary += '-' * 75 + '\n'
    summary += '{0:60}{1:15.0f}\n'.format('Number of observations:', self.N)
    summary += '{0:60}{1:15.0f}\n'.format('Number of covariates:', self.K)
    summary += '{0:60}{1:15.3f}\n'.format('Residual Sum of Squares:', self.rss)
    summary += '{0:60}{1:15.3f}\n'.format('Effective number of parameters (trace(S)):', self.v1)
    summary += '{0:60}{1:15.3f}\n'.format('Degree of freedom (n - trace(S)):', self.df)
    summary += '{0:60}{1:15.3f}\n'.format('Sigma estimate:', self.sigma_hat)
    summary += '{0:60}{1:15.3f}\n'.format('Maximum Log-likelihood:', self.mll)
    summary += '{0:60}{1:15.3f}\n'.format('AIC:', self.aic)
    summary += '{0:60}{1:15.3f}\n'.format('AICc:', self.aicc)
    summary += '{0:60}{1:15.3f}\n'.format('BIC:', self.bic)
    summary += '{0:60}{1:15.3f}\n'.format('R2:', self.R2)
    summary += '{0:60}{1:15.3f}\n'.format('Adj. R2:', self.adjR2)
    summary += '{0:60}{1:15.3f}\n'.format('Adj. alpha (95%):', self.alpha_c)
    summary += '{0:60}{1:15.3f}\n\n'.format('Adj. critical t value (95%):', self.tval_c)
    
    summary += '{}\n'.format('Summary Statistics For GWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    
    summary += '{0:15} {1:>11} {2:>11} {3:>11} {4:>11} {5:>11}\n'.format('Variable',
                                                                         'Mean',
                                                                         'Std',
                                                                         'Min',
                                                                         'Median',
                                                                         'Max')
    summary += '{0:15} {1:11} {2:11} {3:11} {4:11} {5:11}\n'.format('-' * 15,
                                                                    '-' * 11,
                                                                    '-' * 11,
                                                                    '-' * 11,
                                                                    '-' * 11,
                                                                    '-' * 11)
    
    for k in range(self.K):
      summary += '{0:15} {1:11.3f} {2:11.3f} {3:11.3f} {4:11.3f} {5:11.3f}\n'.format(XNames[k],
                                                                                     beta_hat_mean[k],
                                                                                     beta_hat_std[k],
                                                                                     beta_hat_min[k],
                                                                                     beta_hat_median[k],
                                                                                     beta_hat_max[k])
        
    summary += '=' * 75 + '\n'
    summary += '\n'
    
    return summary