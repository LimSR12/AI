import matplotlib.pyplot as plt
import numpy as np

# N값 (행 인덱스)
n_values = [4, 8, 16, 32, 64, 128]

# 각 K에 대한 MSE
mse_k4 =   [976.0208, 414.9012, 295.203, 188.1552, 132.2538, 92.4885]
mse_k16 =  [5287.8964, 2984.2478, 2288.1683, 1665.0287, 1311.6019, 1103.3089]
mse_k64 =  [30420.5242, 21078.8764, 16907.2581, 13373.9274, 11107.163, 9679.2654]

# PSNR 계산 함수
def calculate_psnr(mse_list, max_pixel=255.0):
    return [10 * np.log10((max_pixel ** 2) / mse) for mse in mse_list]

# 각 K에 대한 PSNR 계산
psnr_k4 = calculate_psnr(mse_k4)
psnr_k16 = calculate_psnr(mse_k16)
psnr_k64 = calculate_psnr(mse_k64)

# 그래프 그리기
plt.figure(figsize=(10, 6))

plt.plot(n_values, psnr_k4, marker='o', label='K = 4')
plt.plot(n_values, psnr_k16, marker='s', label='K = 16')
plt.plot(n_values, psnr_k64, marker='^', label='K = 64')

plt.xlabel("N (Number of Clusters)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs N for Different K values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
