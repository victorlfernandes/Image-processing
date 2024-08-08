'''****************************************************************************
|                       USP - Universidade de Sao Paulo                       |
|            ICMC - Instituto de Ciencias Matematicas e de Computacao         |
*******************************************************************************
|                    Bacharelado em Ciencias de Computacao                    |
|                                 2024/1                                      |
|                   SCC0251 - Image Processing and Analysis                   |
|                                                                             |
|              Author: Victor Lucas de Almeida Fernandes (12675399)           |
*******************************************************************************
>               Assignment 3: Color & Segmentation & Morphology
****************************************************************************'''

import numpy as np
import imageio.v3 as imageio
    
def thresholding(f, L):
    f_tr = np.zeros(f.shape, dtype=np.uint8)
    f_tr[f >= L] = 1
    return f_tr

def otsu_threshold(img, max_L):
    M = np.prod(img.shape)
    min_var = []
    hist_t, _ = np.histogram(img, bins=256, range=(0, 256))

    for L in range(1, max_L):
        img_ti = thresholding(img, L)
        indices_a = img_ti == 0
        indices_b = img_ti == 1

        w_a = np.sum(hist_t[:L]) / float(M)
        w_b = np.sum(hist_t[L:]) / float(M)

        if np.any(indices_a) and np.any(indices_b):
            sig_a = np.var(img[indices_a]) if indices_a.any() else 0
            sig_b = np.var(img[indices_b]) if indices_b.any() else 0
            min_var.append(w_a * sig_a + w_b * sig_b)

    optimal_L = np.argmin(min_var) if min_var else 0
    img_t = thresholding(img, optimal_L)

    return img_t

def filter_gaussian(P, Q):
    s1 = P
    s2 = Q

    D = np.zeros([P, Q])  # Compute Distances
    for u in range(P):
        for v in range(Q):
            x = (u-(P/2))**2/(2*s1**2) + (v-(Q/2))**2/(2*s2**2)
            D[u, v] = np.exp(-x)
    return D

def map_value_to_color(value, min_val, max_val, colormap):
    # Scale the value to the range [0, len(colormap) - 1]
    scaled_value = (value - min_val) / (max_val - min_val) * (len(colormap) - 1)
    # Determine the two closest colors in the colormap
    idx1 = int(scaled_value)
    idx2 = min(idx1 + 1, len(colormap) - 1)
    # Interpolate between the two colors based on the fractional part
    frac = scaled_value - idx1
    color = [
        (1 - frac) * colormap[idx1][0] + frac * colormap[idx2][0],
        (1 - frac) * colormap[idx1][1] + frac * colormap[idx2][1],
        (1 - frac) * colormap[idx1][2] + frac * colormap[idx2][2]
    ]
    return color

def rms_error(img, out):
    M,N = img.shape
    error = ((1/(M*N))*np.sum((img-out)**2))**(1/2)
    return error

def main():

    # reading inputs
    input_img_name = str(input().rstrip())
    ref_img_name = str(input().rstrip())
    index = [int(x) for x in input().split()]

    # loading images
    input_img = imageio.imread(input_img_name)
    ref_img = imageio.imread(ref_img_name)

    # trasforming the input img into grayscale
    if len(input_img.shape) > 2:
        gs_img = np.dot(input_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)    
    else:
        gs_img = input_img.copy()
    
    # adaptive limiarization to binarize the image
    bin_img = otsu_threshold(gs_img, 255)

    # applying mathematical morphological operators
    morf_img = bin_img.copy()
    for i in index:
        # erosion
        if i == 1:
            for i in range(1, morf_img.shape[0] - 1):
                for j in range(1, morf_img.shape[1] - 1):
                    if np.min(morf_img[i-1:i+2, j-1:j+2]) == 1:
                        morf_img[i, j] = 1
                    else:
                        morf_img[i, j] = 0
        
        # dilation
        elif i == 2:
            for i in range(1, morf_img.shape[0] - 1):
                for j in range(1, morf_img.shape[1] - 1):
                    if np.max(morf_img[i-1:i+2, j-1:j+2]) == 0:
                        morf_img[i, j] = 0
                    else:
                        morf_img[i, j] = 1


    # applying color to regions of interest
    heatmap_colors = [
        [1, 0, 1],   # Pink
        [0, 0, 1],   # Blue
        [0, 1, 0],   # Green
        [1, 1, 0],   # Yellow
        [1, 0, 0]    # Red
    ]

    alpha = 0.30
    mask = morf_img

    M, N = mask.shape[0], mask.shape[1]
    color_distribution = filter_gaussian(M, N)
    min_val = np.min(np.array(color_distribution))
    max_val = np.max(np.array(color_distribution))

    heatmap_image = np.zeros([M, N, 3]) 
    for i in range(M):
        for j in range(N):
            heatmap_image[i, j] = map_value_to_color(color_distribution[i, j], min_val, max_val, heatmap_colors)

    img_color = np.ones([M, N, 3]) 
    indexes = np.where(mask==0)
    img_color[indexes] = heatmap_image[indexes]

    # normalizing images
    gray_image_normalized = gs_img / np.max(gs_img)
    img_color = (img_color.astype(np.float16) * 255).clip(0, 255).astype(np.uint8)
    gray_image_normalized = (gray_image_normalized * 255).astype(np.uint8)
    gray_image_expanded = np.expand_dims(gray_image_normalized, axis=-1)

    # mixing gray and color image
    mixed_image = (((1 - alpha) * gray_image_expanded).astype(np.float16) + (alpha * img_color.astype(np.float16)).astype(np.float16)).astype(np.uint8)

    # calculating error
    error_R = rms_error(mixed_image[:,:,0], ref_img[:,:,0])
    error_G = rms_error(mixed_image[:,:,1], ref_img[:,:,1])
    error_B = rms_error(mixed_image[:,:,2], ref_img[:,:,2])
    error = (error_R + error_G + error_B)/3
    print(f"{error:.4f}")

if __name__ == '__main__':
  main()