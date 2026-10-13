"""W05. Naive Bayes appropriate to your representation. Implement AT LEAST ONE of the two
classes below and leave the other raising NotImplementedError.

Interface (Appendix A):
    class GaussianNB(var_smoothing=1e-9)       continuous features; per-class variances use ddof = 0
                                               plus var_smoothing * (largest feature variance of X)
    class MultinomialNB(alpha=1.0)             count features, Laplace/Lidstone smoothing
        fit(X, y) -> self; attribute classes_ (sorted)
        predict_log_proba(X) -> (n, K) normalised log-posteriors, columns in classes_ order
        predict(X) -> labels from classes_
"""
import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class MultinomialNB:
    def __init__(self, alpha=1.0):
        # alpha=1.0 là Laplace
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Classes (đã sắp xếp tăng dần), K = số lớp, d = số đặc trưng
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]

        # counts[k, j]: tổng giá trị đặc trưng j của tất cả mẫu thuộc lớp k
        # class_count[k]: số mẫu thuộc lớp k (để tính prior)
        counts = np.zeros((K, d))
        class_count = np.zeros(K)
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]                  # các mẫu thuộc lớp c
            counts[k] = Xc.sum(axis=0)      # cộng dồn theo từng đặc trưng
            class_count[k] = Xc.shape[0]    # đếm số mẫu của lớp

        # Smoothed: cộng alpha vào mỗi feature để xác suất ko bằng 0
        smoothed = counts + self.alpha

        # log P(x_j | lớp k) = log(count_kj + alpha) - log(tổng count_k + alpha*d)
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))

        # log prior của mỗi lớp = log(count của lớp / count của mẫu)
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)

        # Joint log-likelihood: log P(k) + sum_j x_j * log P(x_j | k)  -> shape (n, K)
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_

        # Chuẩn hoá bằng log-sum-exp (trừ max để ổn định số học, tránh overflow/underflow)
        m = jll.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True))

        # log posterior đã chuẩn hoá: mỗi hàng có tổng exp = 1
        return jll - log_norm

    def predict(self, X):
         # Chọn lớp có log-posterior lớn nhất cho mỗi mẫu
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
