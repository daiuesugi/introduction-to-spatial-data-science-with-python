import numpy as np
from .commons import *

class OLS:
  """
    最小二乗法推定を行うクラス
  """
  def __init__(self, y, X):
    # メンバ変数    
    self.y = y  # 被説明変数
    self.X = X  # 説明変数(定数項あり)
    self.N = X.shape[0]  # データサンプル数
    self.K = X.shape[1]  # 係数パラメータ数
    self.df = self.N - self.K  # 自由度
        
    self.beta_hat = None  # 係数パラメータの推定値
    self.S = None  # ハット行列
    self.y_hat = None  # yの理論値
    self.resid = None  # 残差
    self.rss = None  # 残差平方和
    self.sigma2 = None  # 推定された分散パラメータ
    self.sigma_hat = None  # sigma2の平方根
    self.mll = None  # 最大対数尤度
    self.aic = None  # AIC
    self.aicc = None  # AICc
    self.bic = None  # BIC
    self.R2 = None  # 決定係数
    self.adjR2 = None  # 自由度修正済み決定係数
    self.se_beta = None  # 係数パラメータの推定値の標準誤差
    self.t_val = None  # 係数パラメータの推定値のt値
    self.p_val = None  # 係数パラメータの推定値のp値
    
  # 係数パラメータ(beta_hat)を求める関数
  def get_beta_hat(self):
    y = self.y
    X = self.X
    
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    beta_hat = beta_hat.flatten()  # 配列の一次元化
    
    return beta_hat

  # ハット行列(S)を求める関数
  def get_S(self):
    y = self.y
    X = self.X
    
    S = X @ np.linalg.inv(X.T @ X) @ X.T
    
    return S
  
  # モデルを推定する関数(commons.pyにて作成)
  def fit(self):
    self.beta_hat = self.get_beta_hat()
    self.S = self.get_S()
    self.y_hat = get_y_hat(self.y, self.S)
    self.resid = get_resid(self.y, self.y_hat)
    self.rss = get_rss(self.resid)
    self.sigma2 = get_sigma2(self.rss, self.N, self.K)
    self.sigma_hat = np.sqrt(self.sigma2)
    self.se_beta = get_se_beta(self.X, self.sigma2)
    self.t_val = get_t_val(self.beta_hat, self.se_beta)
    self.p_val = get_p_val(self.t_val, self.N, self.K)
    self.mll = get_mll(self.y, self.y_hat, self.rss, self.N)
    self.aic = get_aic(self.mll, self.K)
    self.aicc = get_aicc(self.mll, self.N, self.K)
    self.bic = get_bic(self.mll, self.N, self.K)
    self.R2, self.adjR2 = get_R2(self.y, self.rss, self.N, self.K)
    
  # 結果の要約を表示する関数
  def summary(self):
    XNames = ["X" + str(k) for k in range(self.K)]  # 変数名のラベル
    
    summary = '{}\n'.format('Global Regression Results')
    summary += '=' * 75 + '\n'
    summary += '{0:60}{1:15.0f}\n'.format('Number of observations:', self.N)
    summary += '{0:60}{1:15.0f}\n'.format('Number of covariates:', self.K)
    summary += '{0:60}{1:15.3f}\n'.format('Residual Sum of Squares:', self.rss)
    summary += '{0:60}{1:15.3f}\n'.format('Maximum Log-likelihood:', self.mll)
    summary += '{0:60}{1:15.3f}\n'.format('AIC:', self.aic)
    summary += '{0:60}{1:15.3f}\n'.format('AICc:', self.aicc)
    summary += '{0:60}{1:15.3f}\n'.format('BIC:', self.bic)
    summary += '{0:60}{1:15.3f}\n'.format('R2:', self.R2)
    summary += '{0:60}{1:15.3f}\n\n'.format('Adj. R2:', self.adjR2)
    
    summary += '{0:23} {1:>12} {2:>12} {3:>12} {4:>12}\n'.format('Variable',
                                                                 'Estimate',
                                                                 'Std Error',
                                                                 't-value',
                                                                 'p-value')
    summary += '{0:23} {1:12} {2:12} {3:12} {4:12}\n'.format('-' * 23,
                                                             '-' * 12,
                                                             '-' * 12,
                                                             '-' * 12,
                                                             '-' * 12)
    
    for k in range(self.K):
      summary += '{0:23} {1:12.4g} {2:12.4g} {3:12.3f} {4:12.3f}\n'.format(XNames[k],
                                                                           self.beta_hat[k],
                                                                           self.se_beta[k],
                                                                           self.t_val[k],
                                                                           self.p_val[k])
    
    
    summary += '=' * 75 + '\n'
    summary += '\n'
    
    return summary
    