import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    ans = np.empty((seq_len, d_model), dtype = float)
    # 创建位置向量
    pos = np.arange(seq_len)[:, np.newaxis] # np.arange()等差数列，np.newaxis增加维度
    # 创建频率向量
    k = np.arange(0, d_model, 2) # k = 2i
    freq = np.power(base, k/d_model) # np.power()做幂运算
    # 计算角度
    angle = pos/freq
    # 填充编码矩阵
    ans[:, 0::2] = np.sin(angle) # 偶数列
    # 注意：如果 d_model 是奇数，angles 的列数会比奇数列位置多 1
    # 此时切片 :d_model//2 刚好能取到对应数量的列，避免形状不匹配报错
    ans[:, 1::2] = np.cos(angle[:, :d_model//2])
    return ans
    