import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from src.filters import selective_peeling_filter, fuzzy_weighted_linear_filter
from src.utils import mse, mae, psnr, add_salt_pepper

def run_hybrid_filter(img, peel_iter=2, fuzzy_sigma=10.0):
    # Step 1: Selective Peeling
    step1 = selective_peeling_filter(img, window_size=5, threshold=25, iterations=peel_iter)
    # Step 2: Fuzzy Weighted
    step2 = fuzzy_weighted_linear_filter(step1, window_size=5, sigma=fuzzy_sigma)
    return step2

def main():
    # 1. Load ảnh gốc
    img_path = "data/lena.tif"
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        print(f"Lỗi: Không thể đọc file {img_path}")
        return

    # Nếu muốn chạy nhanh hơn để test, bạn có thể resize ảnh (optional)
    # original = cv2.resize(original, (256, 256))

    print("Đang tính toán các chỉ số cho biểu đồ (sẽ mất vài phút)...")

    # Các mức nhiễu Impulse cần test (từ 5% đến 70%)
    noise_percents = [5, 10, 20, 30, 40, 50, 60, 70]
    
    # Lưu kết quả
    metrics = {
        "mse": {"hybrid": [], "median": []},
        "mae": {"hybrid": [], "median": []},
        "psnr": {"hybrid": [], "median": []}
    }

    for p in noise_percents:
        prob = p / 100.0
        print(f"Processing noise: {p}%...", end=" ", flush=True)
        
        # Tạo ảnh nhiễu
        noisy_img = add_salt_pepper(original, prob=prob)
        
        # --- Filter 1: Proposed Hybrid ---
        # Tăng số lần peel với nhiễu cao để hiệu quả hơn (như bài báo gợi ý)
        iters = 2 if prob < 0.4 else 4
        hybrid_res = run_hybrid_filter(noisy_img, peel_iter=iters)
        
        # --- Filter 2: Standard Median (OpenCV) ---
        median_res = cv2.medianBlur(noisy_img, 5)
        
        # Tính metrics
        metrics["mse"]["hybrid"].append(mse(original, hybrid_res))
        metrics["mse"]["median"].append(mse(original, median_res))
        
        metrics["mae"]["hybrid"].append(mae(original, hybrid_res))
        metrics["mae"]["median"].append(mae(original, median_res))
        
        metrics["psnr"]["hybrid"].append(psnr(original, hybrid_res))
        metrics["psnr"]["median"].append(psnr(original, median_res))
        
        print("Done.")

    # 2. Vẽ biểu đồ (Figure 2 Replication)
    print("Đang vẽ biểu đồ...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Style cho đường
    style_hybrid = {'color': 'black', 'marker': 'o', 'linestyle': '-', 'label': 'Proposed Filter'}
    style_median = {'color': 'red', 'marker': 'x', 'linestyle': '--', 'label': 'Median Filter'}

    # Plot (a) MSE
    ax = axes[0]
    ax.plot(noise_percents, metrics["mse"]["median"], **style_median)
    ax.plot(noise_percents, metrics["mse"]["hybrid"], **style_hybrid)
    ax.set_title('(a) MSE vs. Impulse Percentage')
    ax.set_xlabel('Impulse Noise (%)')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Plot (b) MAE
    ax = axes[1]
    ax.plot(noise_percents, metrics["mae"]["median"], **style_median)
    ax.plot(noise_percents, metrics["mae"]["hybrid"], **style_hybrid)
    ax.set_title('(b) MAE vs. Impulse Percentage')
    ax.set_xlabel('Impulse Noise (%)')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Plot (c) PSNR
    ax = axes[2]
    ax.plot(noise_percents, metrics["psnr"]["median"], **style_median)
    ax.plot(noise_percents, metrics["psnr"]["hybrid"], **style_hybrid)
    ax.set_title('(c) PSNR vs. Impulse Percentage')
    ax.set_xlabel('Impulse Noise (%)')
    ax.set_ylabel('PSNR (dB)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Lưu ảnh
    output_file = "figure_2_replication.png"
    plt.savefig(output_file, dpi=300)
    print(f"Đã lưu biểu đồ tại: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()