import numpy as np
import scipy.stats as st  # 統計分布用モジュール


# yの理論値(y_hat)を求める関数
def get_y_hat(y, S):
    y_hat = S @ y
    
    return y_hat

# 残差(resid)を求める関数
def get_resid(y, y_hat):
    resid = y - y_hat
    
    return resid

# 残差平方和(rss)を求める関数
def get_rss(resid):    
    rss = resid.T @ resid
    
    return rss[0][0]

# 分散パラメータ(sigma2)を求める関数
def get_sigma2(rss, N, K):
    sigma2 = rss / (N - K)
      
    return sigma2

# 係数パラメータの標準誤差(se_beta)を求める関数
def get_se_beta(X, sigma2):
    se_beta = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))

    return se_beta

# 係数パラメータのt値(t_val)を求める関数
def get_t_val(beta_hat, se_beta):
    t_val = beta_hat / se_beta
    
    return t_val

# 係数パラメータのp値(p_val)を求める関数
def get_p_val(t_val, N, K):
    rv = st.t(N - K)
    p_val = 1 - rv.cdf(np.abs(t_val))
    
    return p_val

# 最大対数尤度(mll)を求める関数
def get_mll(y, y_hat, rss, N):
    rv = st.norm(y_hat, np.sqrt(rss/N))
    mll = np.sum(np.log(rv.pdf(y)))
    
    return mll

# AIC(aic)を求める関数
def get_aic(mll, K):
    aic = (-2.0) * mll + 2.0 * K

    return aic

# AICc(aicc)を求める関数
def get_aicc(mll, N, K):
    aicc = (-2.0) * mll + (2.0 * K * N) / (N - K - 1.0)

    return aicc

# BIC(bic)を求める関数
def get_bic(mll, N, K):
    bic = (-2.0) * mll + np.log(N) * K

    return bic

# 決定係数(R2)および自由度調整済み決定係数(adjR2)を求める関数
def get_R2(y, rss, N, K):
    total_var = np.sum((y.flatten() - np.mean(y))**2)
    unexp_var = rss
    
    R2 = 1 - unexp_var/total_var
    adjR2 = 1 - (unexp_var / (N - K)) / (total_var / (N - 1))
    
    return R2, adjR2

# 有効パラメータ数(v1)を求める関数
def get_v1(S):
  v1 = np.trace(S)

  return v1

# 修正alpha(alpha_c)を求める関数
def get_alpha_c(p, v1, xi=0.05):  # p:係数パラメータ数　v1:有効パラメータ数　有意水準0.05
  # 修正alpha(95％)
  alpha_c = xi/(v1/p)
  
  return alpha_c

# 修正臨界t値(tval_c)を求める関数
def get_tval_c(N, p, alpha_c):  # N:データ数　p:係数パラメータ数
  tval_c = st.t.ppf(1-alpha_c/2, N-p) # t分布の自由度は(データ数-係数パラメータ数)

  return tval_c

# local_R2を求める関数
def get_local_R2(y, y_hat, N, w_list):
  # local_R2を格納する配列を作成
  local_R2 = np.zeros(N)
  
  for i in range(N):
    # 重みが対角要素に並んだ「重み行列」の作成
    W_i = np.diag(w_list[i])
    # 全変動
    local_y_bar = np.sum(W_i / np.trace(W_i) @ y)
    total_var_i = (y - local_y_bar).T @ W_i @ (y - local_y_bar)
    # 残差変動
    unexp_var_i = (y - y_hat).T @ W_i @ (y - y_hat)
    # 決定係数
    local_R2_i = 1 - unexp_var_i/ total_var_i
    local_R2[i] = local_R2_i[0,0]
    
  return local_R2

# ローカル係数パラメータの標準誤差(local_std)およびt値(local_t)を求める関数
def get_local_std_and_local_t(X, beta_hat, sigma2, N, w_list):
  # ローカル係数パラメータの標準誤差local_std
  # およびt値local_tを格納する配列を作る
  local_std = np.zeros_like(X)
  local_t = np.zeros_like(X)

  for i in range(N):
    # 重みが対角要素に並んだ「重み行列」の作成
    W_i = np.diag(w_list[i])
    
    # パラメータ推定値の分散共分散行列の対角要素から標準誤差を求める
    C = np.linalg.inv(X.T @ W_i @ X) @ X.T @ W_i
    beta_i_hat_var = sigma2 * C @ C.T
    diag_i = np.diag(beta_i_hat_var).flatten()
    local_std[i, :] = np.sqrt(diag_i)
    
    # t値を求める
    local_t[i, :] = beta_hat[i, :] / local_std[i, :]

  return local_std, local_t

# バンド幅に基づく重みリストを作成する関数
def get_weight_list(N, sorted_dist_list, reverse_dist_list_idx, bw_idx):
  bw = sorted_dist_list[:,bw_idx].reshape([-1, 1])  # バンド幅(bw_idx番目(原点も含む)の列ベクトルの作成)
  w = (1 - (sorted_dist_list / bw)**2)**2  # bi-square型カーネルで重みづけ
  w[:, bw_idx:] = 0  # バンド幅より遠い地点は重みを0とする
  w_list = np.array([w[i, reverse_dist_list_idx[i]] for i in range(N)])  # 順番をもとに戻す

  return w_list

# 重みリストから1行取り出して、重み行列を作成する関数
def get_weight_matrix(w_list):
  W = np.matrix(np.diag(w_list))

  return W