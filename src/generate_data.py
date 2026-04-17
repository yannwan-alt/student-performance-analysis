import pandas as pd
import numpy as np

def generate_data(n=40):
    np.random.seed(42)

    attendance = np.random.normal(75, 15, n).clip(40, 100)
    assignment = np.random.normal(70, 15, n).clip(40, 100)

    final_score = (
        0.5 * attendance +
        0.5 * assignment +
        np.random.normal(0, 5, n)
    ).clip(50, 100)

    # 异常样本（论文亮点）
    attendance[0] = 45
    final_score[0] = 92

    attendance[1] = 95
    final_score[1] = 65

    df = pd.DataFrame({
        "attendance": attendance,
        "assignment": assignment,
        "final_score": final_score
    })

    return df