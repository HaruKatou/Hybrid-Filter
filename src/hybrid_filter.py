import numpy as np
from numba import jit

def selective_peeling_filter_optimized(img, window_size=3, threshold=30, iterations=1):
    """
    Step 1: Nonlinear Impulse Removal
    Removes outliers (impulses) by peeling from Max/Min towards Median.
    """
    pad = window_size // 2
    h, w = img.shape
    # Work on a copy to avoid race conditions during iteration
    img_filtered = img.copy().astype(np.int32)
    
    for it in range(iterations):
        # Create a buffer for this iteration's updates
        new_img = img_filtered.copy()
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                # 1. Extract Window
                # Numba doesn't support advanced slicing well, so we do it manually or simple slice
                win_flat = img_filtered[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
                
                # 2. Sort Window
                # Quicksort is default in numpy, usually fast enough
                win_sorted = np.sort(win_flat)
                n = len(win_sorted)
                median_idx = n // 2
                median = win_sorted[median_idx]
                center_val = img_filtered[i, j]
                
                # 3. Peeling Logic (Checking BOTH ends)
                # Check if the center pixel is an outlier
                is_outlier = False
                
                # Check Maximum side (Right to Left)
                # We only care if the current center_val is one of the peeled values
                for k in range(n - 1, median_idx, -1):
                    val = win_sorted[k]
                    if abs(val - median) > threshold:
                        # This value is an outlier. Is it our center pixel?
                        if center_val == val:
                            is_outlier = True
                            break
                    else:
                        # Stop peeling this side if we hit a 'good' pixel
                        break
                
                # Check Minimum side (Left to Right) - Only if not already found
                if not is_outlier:
                    for k in range(0, median_idx):
                        val = win_sorted[k]
                        if abs(val - median) > threshold:
                            if center_val == val:
                                is_outlier = True
                                break
                        else:
                            break
                
                # 4. Replacement
                if is_outlier:
                    new_img[i, j] = median
                    
        img_filtered = new_img

    return img_filtered.astype(np.uint8)

@jit(nopython=True)
def fuzzy_weighted_linear_filter_optimized(img, window_size=5, sigma=10.0):
    """
    Step 2: Linear Filter with Fuzzy Weights
    Weighted average where weights depend on intensity difference from center.
    """
    pad = window_size // 2
    h, w = img.shape
    img_float = img.astype(np.float32)
    filtered = np.zeros_like(img_float)
    
    # Precompute Gaussian lookup table (Fuzzy Membership)
    # Range 0-255
    fuzzy_table = np.exp(-(np.arange(256)**2) / (2 * sigma**2))
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            
            center_val = img_float[i, j]
            numerator = 0.0
            denominator = 0.0
            
            # Convolve manually for speed in Numba
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    neighbor_val = img_float[i + m, j + n]
                    
                    # Calculate Difference (Delta x)
                    diff = int(abs(neighbor_val - center_val))
                    
                    # Lookup Fuzzy Weight
                    weight = fuzzy_table[diff]
                    
                    numerator += weight * neighbor_val
                    denominator += weight
            
            # Avoid division by zero
            if denominator > 0:
                filtered[i, j] = numerator / denominator
            else:
                filtered[i, j] = center_val

    return filtered.astype(np.uint8)