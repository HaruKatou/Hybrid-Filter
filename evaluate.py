import cv2
import numpy as np
import pandas as pd
import time
from src.filters import selective_peeling_filter, fuzzy_weighted_linear_filter
from src.utils import mse, mae, psnr, add_salt_pepper, add_gaussian_noise

def run_hybrid_filter(img, peel_iter=2, fuzzy_sigma=10.0):
    # Bước 1: Bộ lọc phi tuyến (Peeling) để loại bỏ nhiễu xung
    # Threshold=25 và Iterations=2 là tham số khuyến nghị cho ảnh nhiều nhiễu
    step1 = selective_peeling_filter(img, window_size=5, threshold=25, iterations=peel_iter)
    
    # Bước 2: Bộ lọc tuyến tính (Fuzzy) để loại bỏ nhiễu Gaussian còn sót và làm mịn
    step2 = fuzzy_weighted_linear_filter(step1, window_size=5, sigma=fuzzy_sigma)
    return step2

def evaluate_table_2(original_img):
    """
    Tái tạo Table 2: So sánh MSE, MAE, PSNR theo % nhiễu xung (Impulse Noise).
    Chạy từ 5% đến 60% như bài báo.
    """
    print("\n" + "="*60)
    print("GENERATING TABLE 2: Performance vs Impulse Noise Density")
    print("Note: Chạy trên ảnh gốc, vui lòng đợi...")
    print("="*60)
    
    # Các mức nhiễu cần test (tương tự Table 2 trong bài báo)
    noise_levels = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    results = []

    for p in noise_levels:
        print(f"Processing noise level: {int(p*100)}%...", end=" ", flush=True)
        start_t = time.time()
        
        # 1. Tạo ảnh nhiễu muối tiêu
        noisy_img = add_salt_pepper(original_img, prob=p)
        
        # 2. Chạy Filter của BẠN (Proposed Hybrid)
        # Với nhiễu cao (>30%), tăng số lần peeling lên để hiệu quả hơn
        iters = 2 if p < 0.4 else 3
        hybrid_res = run_hybrid_filter(noisy_img, peel_iter=iters)
        
        # 3. Chạy Median Filter (OpenCV chuẩn) để so sánh
        median_res = cv2.medianBlur(noisy_img, 5)
        
        elapsed = time.time() - start_t
        print(f"Done ({elapsed:.1f}s)")
        
        # 4. Tính Metrics
        res_row = {
            "Noise %": f"{int(p*100)}%",
            "MSE (Ours)": f"{mse(original_img, hybrid_res):.1f}",
            "MSE (Med)": f"{mse(original_img, median_res):.1f}",
            "MAE (Ours)": f"{mae(original_img, hybrid_res):.2f}",
            "MAE (Med)": f"{mae(original_img, median_res):.2f}",
            "PSNR (Ours)": f"{psnr(original_img, hybrid_res):.2f}",
            "PSNR (Med)": f"{psnr(original_img, median_res):.2f}",
        }
        results.append(res_row)

    df = pd.DataFrame(results)
    print("\n=== KẾT QUẢ TABLE 2 (Tái tạo) ===")
    print(df.to_string(index=False))
    return df

def evaluate_table_1(original_img):
    """
    Tái tạo Table 1: So sánh các bộ lọc trong môi trường nhiễu hỗn hợp.
    Mô hình: Gaussian (sigma=10) + Impulsive (chọn 20% làm mẫu).
    """
    print("\n" + "="*60)
    print("GENERATING TABLE 1: Mixed Noise Comparison")
    print("Scenario: Gaussian(sigma=10) + Salt&Pepper(20%)")
    print("="*60)
    
    g_sigma = 10
    impulse_prob = 0.20
    
    # Tạo nhiễu hỗn hợp: Cộng Gaussian trước, sau đó rải muối tiêu
    temp = add_gaussian_noise(original_img, mean=0, sigma=g_sigma)
    noisy_mixed = add_salt_pepper(temp, prob=impulse_prob)
    
    # 1. Metrics ảnh nhiễu
    m_noisy = [mse(original_img, noisy_mixed), mae(original_img, noisy_mixed), psnr(original_img, noisy_mixed)]

    # 2. Average Filter (Linear) - đại diện cho Linear Filters
    print("Running Average Filter...")
    avg_res = cv2.blur(noisy_mixed, (5,5))
    m_avg = [mse(original_img, avg_res), mae(original_img, avg_res), psnr(original_img, avg_res)]

    # 3. Median Filter (Nonlinear) - đại diện cho Nonlinear Filters
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
    print("\n=== KẾT QUẢ TABLE 1 (Tái tạo) ===")
    print(df.to_string(index=False))

def main():
    # Đọc ảnh (Đảm bảo đường dẫn đúng)
    img_path = "data/lena.tif" 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}. Hãy kiểm tra lại đường dẫn.")
        return

    print(f"Đã load ảnh: {img_path} với kích thước gốc: {img.shape}")
    
    # Xử lý ảnh 16-bit nếu cần
    if img.dtype == 'uint16':
        img = (img / 256).astype('uint8')

    # Chạy đánh giá
    evaluate_table_2(img)
    evaluate_table_1(img)

if __name__ == "__main__":
    main()