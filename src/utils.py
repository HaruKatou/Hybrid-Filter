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
