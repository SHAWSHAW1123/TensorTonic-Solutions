def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    for step in range(steps):
        #计算导数（梯度）
        dy = 2*a*x0 + b
        x0 = x0 - dy * lr
    return x0