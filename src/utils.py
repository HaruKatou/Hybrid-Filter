import numpy as np

def mse(original, filtered):
    return np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)

def mae(original, filtered):
    return np.mean(np.abs(original.astype(np.float32) - filtered.astype(np.float32)))

def psnr(original, filtered):
    m = mse(original, filtered)
    return 10 * np.log10((255 ** 2) / m)

def add_salt_pepper(img, prob):
    noisy = img.copy()
    mask = np.random.rand(*img.shape)
    noisy[mask < prob/2] = 0
    noisy[(mask >= prob/2) & (mask < prob)] = 255
    return noisy

def add_gaussian_noise(img, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_image = img.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# # Test utility functions
# probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# for p in probs:
#     noisy = add_salt_pepper(original, p)
#     filtered = selective_peeling_filter(noisy)
#     filtered = fuzzy_weighted_linear_filter(filtered)

#     print(f"Noise {p*100:.1f}%:")
#     print("  MSE:", mse(original, filtered))
#     print("  MAE:", mae(original, filtered))
#     print("  PSNR:", psnr(original, filtered))
