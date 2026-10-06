"""W04. Multilayer perceptron with an explicit forward and backward pass.

Interface (Appendix A). params is a dict {"W1", "b1", ..., "WL", "bL"}; W_l has shape
(n_in, n_out). The output layer is softmax; the loss is mean cross-entropy. The hidden
activation is your choice (state it in MODEL_LOG.md).
    init_params(sizes, seed) -> params       sizes = [d, h1, ..., K]
    forward(params, X) -> (P, cache)          P has shape (n, K), rows sum to 1
    loss(params, X, Y) -> float               Y is one-hot (n, K); mean cross-entropy
    backward(params, cache, Y) -> grads       same keys and shapes as params, gradient of loss
"""
import numpy as np


def init_params(sizes, seed=0):
    # RNG có seed để tái lập kết quả
    rng = np.random.default_rng(seed)
    params = {}
    # L = len(sizes) - 1 layer, đánh số từ 1
    for l in range(1, len(sizes)):
        n_in, n_out = sizes[l - 1], sizes[l]
        
        # W_l: shape (n_in, n_out), std = sqrt(1/n_in) (Xavier) để tanh/softmax không bão hòa
        params[f"W{l}"] = rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out))
       
        # b_l: shape (n_out,), khởi tạo = 0
        params[f"b{l}"] = np.zeros(n_out)
    return params


def forward(params, X):
    # số layer: mỗi layer có W và b
    L = len(params) // 2
    A = [np.asarray(X, dtype=float)]  # A[0] = X, A[l] = activation của layer l

    # Hidden layers 1..L-1: Z = A_prev @ W + b, rồi tanh
    for l in range(1, L):
        Z = A[-1] @ params[f"W{l}"] + params[f"b{l}"]
        A.append(np.tanh(Z))

    # Output layer L: logits (n, K), chưa qua activation
    logits = A[-1] @ params[f"W{L}"] + params[f"b{L}"]

    # Softmax ổn định số: trừ max từng hàng để exp không tràn
    logits = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(logits)
    
    # (n, K), mỗi hàng tổng bằng 1
    P = expz / expz.sum(axis=1, keepdims=True)  

    # cache giữ A[0..L-1] (cho dW và đạo hàm tanh) và P (cho gradient ở output)
    return P, {"A": A, "P": P}


def loss(params, X, Y):
    # tính lại từ params hiện tại, không dùng cache cũ
    P, _ = forward(params, X)  
    n = P.shape[0]
    # Mean cross-entropy: -(1/n) * sum_{i,k} Y_ik * log(P_ik)
    # clip để P = 0 không cho log(0) = -inf
    return float(-np.sum(Y * np.log(np.clip(P, 1e-12, 1.0))) / n)


def backward(params, cache, Y):
    L = len(params) // 2
    A, P = cache["A"], cache["P"]
    n = P.shape[0]
    grads = {}

    # Gradient của mean cross-entropy theo logit (softmax + CE): (P - Y) / n
    dZ = (P - Y) / n

    for l in range(L, 0, -1):  # đi ngược từ layer L về layer 1
        grads[f"W{l}"] = A[l - 1].T @ dZ   # (n_in, n_out)
        grads[f"b{l}"] = dZ.sum(axis=0)    # cộng dồn theo trục mẫu -> (n_out,)

        if l > 1:
            dA = dZ @ params[f"W{l}"].T            # truyền ngược qua W_l
            dZ = dA * (1.0 - A[l - 1] ** 2)        # đạo hàm tanh: 1 - tanh^2

    return grads
