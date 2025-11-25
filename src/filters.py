import numpy as np

def selective_peeling_filter(img, window_size=3, threshold=30, iterations=1):
    """
    Step 1: Selective-Peeling to remove impulsive noise.
    img: grayscale image (numpy array)
    window_size: size of window (must be odd)
    threshold: distance from median to consider pixel an outlier
    iterations: number of peeling passes
    """
    assert window_size % 2 == 1, "window_size must be odd"

    pad = window_size // 2
    h, w = img.shape
    img_filtered = img.copy().astype(np.int32)

    for it in range(iterations):
        new_img = img_filtered.copy()

        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                # Extract local window
                win = img_filtered[i-pad:i+pad+1, j-pad:j+pad+1]

                # Sort to find median
                win_sorted = np.sort(win, axis=None)
                median = win_sorted[len(win_sorted) // 2]

                is_outlier = False

                # Peel from max → downwards
                for val in reversed(win_sorted):
                    if abs(val - median) > threshold:
                        # Replace pixel by median
                        if img_filtered[i, j] == val:
                            new_img[i, j] = median
                            is_outlier = True
                            break
                    else:
                        break

                # Peel from min → upwards
                if not is_outlier:
                    for val in win_sorted:
                        if abs(val - median) > threshold:
                            if img_filtered[i, j] == val:
                                is_outlier = True
                                break
                        else:
                            break
                
                if is_outlier:
                    new_img[i, j] = median

        img_filtered = new_img

    return img_filtered.astype(np.uint8)

def fuzzy_weighted_linear_filter(img, window_size=5, sigma=10.0):
    assert window_size % 2 == 1, "window_size must be odd"

    pad = window_size // 2
    h, w = img.shape
    img = img.astype(np.float32)

    filtered = np.zeros_like(img)

    # Precompute gaussian fuzzy weights for speed
    max_diff = 255
    diff_range = np.arange(0, max_diff + 1)
    fuzzy_table = np.exp(-(diff_range**2) / (2 * sigma**2))

    # Pad image
    padded = np.pad(img, pad, mode='reflect')

    for i in range(h):
        for j in range(w):

            # Extract window
            win = padded[i:i+window_size, j:j+window_size]

            center = padded[i+pad, j+pad]
            diff = np.abs(win - center).astype(np.int32)

            # Lookup fuzzy weights
            weights = fuzzy_table[diff]

            # Weighted average
            numerator = np.sum(weights * win)
            denominator = np.sum(weights)

            filtered[i, j] = numerator / denominator

    return filtered.astype(np.uint8)