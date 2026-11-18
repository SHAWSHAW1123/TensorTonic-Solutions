import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.zeros((0,0), dtype = int)
        
    L = max_len if max_len else max(len(seq) for seq in seqs)

    rows = [] #存储复制的列表
    for seq in seqs:
        row = seq[:L] #利用切片截断
        row.extend([pad_value] * (L - len(seq))) #利用extend填充，extend内必须是可迭代对象，比如列表
        rows.append(row)
        
    ans = np.array(rows)
    return ans
    pass