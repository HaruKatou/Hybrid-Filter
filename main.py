from src.filters import selective_peeling_filter
from src.filters import fuzzy_weighted_linear_filter
from src.utils import mse, mae, psnr
import cv2
import numpy as np

def main():
    img = cv2.imread("data/lena.tif", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Không thể đọc file TIFF!")

    if img.dtype == 'uint16':
        img = (img / 256).astype('uint8')

    step1 = selective_peeling_filter(
        img, 
        window_size=5, 
        threshold=25, 
        iterations=2
    )
    changed1 = np.sum(step1 != img)
    print("Pixels changed:", changed1)
    cv2.imwrite("data/cleaned_step1.tif", step1)

    step2 = fuzzy_weighted_linear_filter(
        step1, 
        window_size=5, 
        sigma=10.0
    )

    changed2 = np.sum(step2 != step1)
    print("Pixels changed:", changed2)

    cv2.imwrite("data/cleaned_step2.tif", step2)

    original = cv2.imread("data/lena.tif", cv2.IMREAD_GRAYSCALE)
    filtered = cv2.imread("data/cleaned_step2.tif", cv2.IMREAD_GRAYSCALE)

    print("MSE:", mse(original, filtered))
    print("MAE:", mae(original, filtered))
    print("PSNR:", psnr(original, filtered))


if __name__ == "__main__":
    main()
