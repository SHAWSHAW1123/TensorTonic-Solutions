import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    #计算梯度，为同步更新梯度，将其打包成一个函数
    def gradient(X,y,w,b):
        h = _sigmoid(X@w + b)
        error = h - y
        n = X.shape[0]
        dw = 1/n * (X.T @ error)
        db = 1/n * (np.sum(error))
        return dw,db

    #训练模型
    m = X.shape[1]
    w = np.zeros(m)
    b = 0.0
    for step in range(steps):
        dw, db = gradient(X,y,w,b)
        w = w - lr * dw
        b = b- lr * db
    return w,b
        
    pass