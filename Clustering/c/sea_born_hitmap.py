import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 정의
data = np.array([
    [976.0208, 5287.8964, 30420.5242],
    [414.9012, 2984.2478, 21078.8764],
    [295.203,  2288.1683, 16907.2581],
    [188.1552, 1665.0287, 13373.9274],
    [132.2538, 1311.6019, 11107.163],
    [92.4885,  1103.3089, 9679.2654]
])

n_values = [4, 8, 16, 32, 64, 128]
k_values = [4, 16, 64]

# Pandas DataFrame 생성 (Seaborn에 더 적합)
df = pd.DataFrame(data, index=n_values, columns=k_values)

# Heatmap 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("MSE Heatmap by (N, K)")
plt.xlabel("K")
plt.ylabel("N")
plt.tight_layout()
plt.show()
