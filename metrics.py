import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils import to_numpy

def sample_sphere(n, p, random_state):
    rng = np.random.default_rng(random_state)
    v = rng.normal(0, 1, size=(p, n))
    v /= np.linalg.norm(v, axis=0)
    return v.T

def wsc_unbiased(X, covered, eta=0.2, M=1000, test_size=0.75, random_state=42):
    """Computes Worst-Slice Coverage."""
    if len(X) < 100: return 0.0 
    X_train, X_test, cov_train, cov_test = train_test_split(X, covered, test_size=test_size, random_state=random_state)
    n = len(X_train)
    V = sample_sphere(M, X_train.shape[1], random_state)
    z = np.dot(X_train, V.T)
    z_order = np.argsort(z, axis=0)
    cover_ordered = np.take_along_axis(cov_train[:, None], z_order, axis=0)
    
    k_min = int(math.ceil(eta * n))
    cum_cov = np.vstack([np.zeros((1, M)), np.cumsum(cover_ordered, axis=0)])
    roll_cov = (cum_cov[k_min:] - cum_cov[:-k_min]) / k_min
    
    min_cov_idx = np.argmin(roll_cov, axis=0)
    idx_star = np.argmin(roll_cov[min_cov_idx, np.arange(M)])
    
    v_star = V[idx_star]
    start_idx = min_cov_idx[idx_star]
    z_sorted = np.take_along_axis(z, z_order, axis=0)
    a_star = z_sorted[start_idx, idx_star]
    b_star = z_sorted[start_idx + k_min - 1, idx_star]
    
    z_test = np.dot(X_test, v_star)
    valid = (z_test >= a_star) & (z_test <= b_star)
    if np.sum(valid) == 0: return 1.0
    return np.mean(cov_test[valid])

class ConditionalCoverageComputer:
    """Computes Conditional Coverage Error (CCE) via partition-based approximation."""
    def __init__(self, X, nb_partitions=10, random_state=42):
        self.kmeans = KMeans(n_clusters=nb_partitions, n_init='auto', random_state=random_state).fit(X)
        self.labels = self.kmeans.labels_
        self.nb = nb_partitions

    def compute_error(self, coverages, alpha):
        err = 0
        total_len = len(coverages)
        for i in range(self.nb):
            mask = self.labels == i
            c = np.sum(mask)
            if c > 0: 
                w = c / total_len
                err += w * (np.mean(coverages[mask]) - (1-alpha))**2
        return err

def get_metrics_nd(y_test, y_lo, y_hi, x_test, alpha=0.1):
    """Calculates all metrics for N-dimensional regression."""
    y_test, y_lo, y_hi, x_test = map(lambda v: to_numpy(v), [y_test, y_lo, y_hi, x_test])
    
    # 1. Marginal Coverage
    in_bounds = (y_test >= y_lo) & (y_test <= y_hi)
    covered = np.all(in_bounds, axis=1)
    cov = np.mean(covered)
    
    # 2. Size (Log Volume approximation)
    D = y_test.shape[1]
    widths = np.maximum(y_hi - y_lo, 1e-6)
    if D > 1: 
        size_metric = np.mean(np.mean(np.log(widths), axis=1))
    else: 
        size_metric = np.mean(widths)
    
    # 3. WSC
    wsc_val = wsc_unbiased(x_test, covered)
    
    # 4. CCE
    cce_computer = ConditionalCoverageComputer(x_test, nb_partitions=50)
    cce_val = cce_computer.compute_error(covered, alpha)
    
    return {"Cov": cov, "Size": size_metric, "WSC": wsc_val, "CCE": cce_val}