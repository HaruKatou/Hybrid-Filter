import cv2
import numpy as np
import pandas as pd
import time
from src.filters import selective_peeling_filter, fuzzy_weighted_linear_filter
from src.utils import mse, mae, psnr, add_salt_pepper, add_gaussian_noise
import matplotlib.pyplot as plt

def run_hybrid_filter(img, peel_iter=2, fuzzy_sigma=130.0):
    step1 = selective_peeling_filter(img, window_size=5, threshold=30, iterations=peel_iter)
    
    step2 = fuzzy_weighted_linear_filter(step1, window_size=5, sigma=fuzzy_sigma)
    return step2

def evaluate_1(original_img):
    print("Đang tính toán các chỉ số cho biểu đồ (sẽ mất vài phút)...")

    noise_percents = [5, 10, 20, 30, 40, 50, 60, 70]

    g_sigma = 10
    
    # Lưu kết quả
    metrics = {
        "mse": {"hybrid": [], "median": []},
        "mae": {"hybrid": [], "median": []},
        "psnr": {"hybrid": [], "median": []}
    }

    for p in noise_percents:
        prob = p / 100.0
        print(f"Processing noise: {p}%...", end=" ", flush=True)
        
        noisy_img = add_gaussian_noise(original_img, mean=0, sigma=g_sigma)
        noisy_img = add_salt_pepper(noisy_img, prob=prob)
        
        # --- Filter 1: Proposed Hybrid ---
        # Tăng số lần peel với nhiễu cao để hiệu quả hơn (như bài báo gợi ý)
        iters = 2 if prob < 0.4 else 4
        hybrid_res = run_hybrid_filter(noisy_img, peel_iter=iters)
        
        # --- Filter 2: Standard Median (OpenCV) ---
        median_res = cv2.medianBlur(noisy_img, 5)
        
        # Tính metrics
        metrics["mse"]["hybrid"].append(mse(original_img, hybrid_res))
        metrics["mse"]["median"].append(mse(original_img, median_res))
        
        metrics["mae"]["hybrid"].append(mae(original_img, hybrid_res))
        metrics["mae"]["median"].append(mae(original_img, median_res))
        
        metrics["psnr"]["hybrid"].append(psnr(original_img, hybrid_res))
        metrics["psnr"]["median"].append(psnr(original_img, median_res))
        
        print("Done.")

    print("Đang vẽ biểu đồ...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    output_file = "figure_2_replication.png"
    plt.savefig(output_file, dpi=300)
    print(f"Đã lưu biểu đồ tại: {output_file}")
    plt.show()

def evaluate_2(original_img):
    """
    Tái tạo Table 1: So sánh các bộ lọc trong môi trường nhiễu hỗn hợp.
    Mô hình: Gaussian (sigma=10) + Impulsive (chọn 20% làm mẫu).
    """
    print("\n" + "="*60)
    print("GENERATING TABLE 1: Mixed Noise Comparison")
    print("Scenario: Gaussian(sigma=10) + Salt&Pepper")
    print("="*60)
    
    g_sigma = 10
    impulse_prob = 0.4
    
    # Tạo nhiễu hỗn hợp: Cộng Gaussian trước, sau đó rải muối tiêu
    temp = add_gaussian_noise(original_img, mean=0, sigma=g_sigma)
    noisy_mixed = add_salt_pepper(temp, prob=impulse_prob)
    
    m_noisy = [mse(original_img, noisy_mixed), mae(original_img, noisy_mixed), psnr(original_img, noisy_mixed)]

    # 2. Average Filter (Linear)
    print("Running Average Filter...")
    avg_res = cv2.blur(noisy_mixed, (5,5))
    m_avg = [mse(original_img, avg_res), mae(original_img, avg_res), psnr(original_img, avg_res)]

    # 3. Median Filter (Nonlinear)
    print("Running Median Filter...")
    med_res = cv2.medianBlur(noisy_mixed, 5)
    m_med = [mse(original_img, med_res), mae(original_img, med_res), psnr(original_img, med_res)]

    # 4. Proposed Hybrid Filter
    print("Running Your Hybrid Filter (Full Resolution)...")
    prop_res = run_hybrid_filter(noisy_mixed, peel_iter=2)
    m_prop = [mse(original_img, prop_res), mae(original_img, prop_res), psnr(original_img, prop_res)]

    # Tạo bảng
    data = {
        "Metric": ["MSE", "MAE", "PSNR"],
        "Noisy Img": [f"{x:.2f}" for x in m_noisy],
        "Ave Filter": [f"{x:.2f}" for x in m_avg],
        "Med Filter": [f"{x:.2f}" for x in m_med],
        "OUR FILTER": [f"{x:.2f}" for x in m_prop]
    }
    
    df = pd.DataFrame(data)
    print("\n=== KẾT QUẢ TABLE 1 ===")
    print(df.to_string(index=False))


def evaluate_3(original_img):
    fig, axes = plt.subplots(1, 5, figsize=(10, 6))

    # Gaussian (0, 10^2) + Salt & Pepper Noise
    g_sigma = 10
    impulse_prob = 0.4

    noisy_img = add_gaussian_noise(original_img, mean=0, sigma=g_sigma)
    noisy_img = add_salt_pepper(noisy_img, prob=impulse_prob)

    # Filter 1: Proposed 
    proposed_res = run_hybrid_filter(noisy_img, peel_iter=2)

    # Filter 2: Median
    median_res = cv2.medianBlur(noisy_img, 5)

    # Filter 3: Average
    average_res = cv2.blur(noisy_img, (5,5))

    output_img = original_img

    axes[0].imshow(output_img, cmap='gray')
    axes[0].set_title("Original Lena Image")
    
    axes[1].imshow(noisy_img, cmap='gray')
    axes[1].set_title(f"Noisy\nMSE={mse(original_img, noisy_img):.4f}")
    
    axes[2].imshow(proposed_res, cmap='gray')
    axes[2].set_title(f"Proposed\nMSE={mse(original_img, proposed_res):.4f}")

    axes[3].imshow(median_res, cmap='gray')
    axes[3].set_title(f"Median\nMSE={mse(original_img, median_res):.4f}")
    
    axes[4].imshow(average_res, cmap='gray')
    axes[4].set_title(f"Average\nMSE={mse(original_img, average_res):.4f}")

    for ax in axes: ax.axis('off')
    plt.tight_layout()

    output_file = "figure_3_replication.png"
    plt.savefig(output_file, dpi=300)
    print(f"Đã lưu biểu đồ tại: {output_file}")
    plt.show()

def main():
    np.random.seed(42)

    img_path = "data/lena.tif" 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}. Hãy kiểm tra lại đường dẫn.")
        return

    print(f"Đã load ảnh: {img_path} với kích thước gốc: {img.shape}")
    
    if img.dtype == 'uint16':
        img = (img / 255.0).astype('uint8')

    evaluate_2(img)
    evaluate_1(img)
    evaluate_3(img)

if __name__ == "__main__":
    main()