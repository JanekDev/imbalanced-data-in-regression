import numpy as np
from math import ceil
from scipy.linalg import qr

def weighted_ls(X, y, w=[]):
    if w == []:
        w = np.ones(len(y),)
    ws = np.sqrt(w)
    WX = ws[:, np.newaxis] * X
    if len(y.shape) > 1:
        wy = ws[:, np.newaxis] * y
    else:
        wy = ws * y
    beta = np.linalg.inv(WX.T @ WX) @ (WX.T @ wy)
    return beta

def detect_outliers(res, corrupt_frac):

    n = res.shape[0]
    outlier_indicator = np.zeros((n,), dtype=np.bool8)
    M = np.min(res, axis=1)
    if corrupt_frac > 0:
        outlier_supp = np.argpartition(M, -round(corrupt_frac*n))[-round(corrupt_frac*n):]
        outlier_indicator[outlier_supp] = 1
   
    return outlier_indicator

def cluster_by_beta(beta, X, y, corrupt_frac):
    K = beta.shape[1]
    res = np.zeros((len(y), K))
    for k in range(K):
        res[:, k] = abs(X @ beta[:, k] - y)
    outlier_indicator = detect_outliers(res, corrupt_frac)
    I = np.argmin(res, axis=1)
    I[outlier_indicator.astype(np.int8)] = -1
    return I

def OLS(X, y, K, c):
    beta_hat = np.zeros((X.shape[1], K))
    for k in range(K):
        beta_hat[:,k] = weighted_ls(X[c==k,:], y[c==k])
    return beta_hat

def find_component(X, y, wfun, nu, rho, iterlim_inner, beta_init, verbose):
    # INPUT:
    # wfun = IRLS reweighting function
    # nu = tuning parameter used in IRLS reweighting function
    # rho = minimal oversampling to detect component
    # iterlim_inner = max inner iters
    # beta_init - initialization
    # OUTPUT:
    # beta = regression over large weights
    # w = final weights
    # iter = inner iters done
    
    d = X.shape[1]
    _, w, iter = MixIRLS_inner(X, y, wfun, nu, False, iterlim_inner, beta_init)
    I = np.argpartition(w, -ceil(rho * d))[-ceil(rho*d):]
    I_count = np.count_nonzero(I)
    beta = weighted_ls(X[I,:], y[I])
    if verbose:
        print('observed error: ' + str(np.linalg.norm(X[I,:] @ beta - y[I]) / np.linalg.norm(y[I])) + '. active support size: ' + str(I_count))
    return beta, w, iter


def MixIRLS_inner(X, y, wfun, nu, intercept, iterlim, beta_init=[]):
    # if beta_init is not supplied or == -1, the OLS is used
    
    n,d = X.shape
    if intercept:
        X = np.c_[np.ones(n,), X]
        d = d+1
    assert n >= d, 'not enough data'

    beta = np.zeros((d,))
    Q, R, perm = qr(X, mode='economic', pivoting=True)
    if beta_init == []:
        beta[perm] = weighted_ls(R, Q.T @ y)
    else:
        beta = beta_init

    # adjust residuals according to DuMouchel & O'Brien (1989)
    E = weighted_ls(R.T, X[:, perm].T).T
    h = np.sum(E * E, axis=1)
    h[h > 1 - 1e-4] = 1 - 1e-4
    adjfactor = 1 / np.sqrt(1-h)

    # IRLS
    for iter in range(iterlim):
        # residuals
        r = adjfactor * (y - X @ beta)
        rs = np.sort(np.abs(r))
        # scale
        s = np.median(rs[d:]) / 0.6745 # mad sigma
        s = max(s, 1e-6 * np.std(y)) # lower bound s in case of a good fit
        if s == 0: # perfect fit
            s = 1
        # weights
        w = wfun(r / (nu * s))
        # beta
        beta_prev = beta.copy()
        beta[perm] = weighted_ls(X[:,perm], y, w)

        # early stop if beta doesn't change
        if np.all(np.abs(beta-beta_prev) <= np.sqrt(1e-16) * np.maximum(np.abs(beta), np.abs(beta_prev))):
            break

    return beta, w, iter