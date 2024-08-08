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
>        Assignment 2: Fourier Transform & Filtering in Frequency Domain
****************************************************************************'''

import numpy as np
import imageio.v3 as imageio

def RMSE(img, ref):

  N = img.shape[0]

  aux = np.float32(0.0)
  for i in range(N):
    for j in range(N):
      aux += np.power(ref[i, j] - img[i, j], 2).astype(np.float32)

  return np.sqrt(aux / (N * N)).astype(np.float32)

def low_pass(input_img, ref_img, radius):

    # generating the frequency domain
    freq_domain = np.fft.fftshift(np.fft.fft2(input_img))

    # generating the filter
    filter = freq_domain.copy()

    P, Q = freq_domain.shape
    for u in range(P):
      for v in range(Q):
        D = np.sqrt(np.power(u - P / 2, 2) + np.power(v - Q / 2, 2))
        if D <= radius:
          filter[u, v] = 1
        else:
          filter[u, v] = 0

    # filtering
    filtered_img = np.multiply(freq_domain, filter)

    # restoring and normalizing image
    restored_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    restored_img = restored_img.real
    restored_img = (restored_img-np.min(restored_img))*(255/(np.max(restored_img)-np.min(restored_img)))

    # comparing with ref
    print(RMSE(restored_img, ref_img))

def high_pass(input_img, ref_img, radius):

    # generating the frequency domain
    freq_domain = np.fft.fftshift(np.fft.fft2(input_img))

    # generating the filter
    filter = freq_domain.copy()

    P, Q = freq_domain.shape
    for u in range(P):
      for v in range(Q):
        D = np.sqrt(np.power(u - P / 2, 2) + np.power(v - Q / 2, 2))
        if D <= radius:
          filter[u, v] = 0
        else:
          filter[u, v] = 1

    # filtering
    filtered_img = np.multiply(freq_domain, filter)

    # restoring and normalizing image
    restored_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    restored_img = restored_img.real
    restored_img = (restored_img-np.min(restored_img))*(255/(np.max(restored_img)-np.min(restored_img)))

    # comparing with ref
    print(RMSE(restored_img, ref_img))

def band_stop(input_img, ref_img, r0, r1):

    # generating the frequency domain
    freq_domain = np.fft.fftshift(np.fft.fft2(input_img))

    # generating the filter
    filter = freq_domain.copy()

    P, Q = freq_domain.shape
    for u in range(P):
      for v in range(Q):
        D = np.sqrt(np.power(u - P / 2, 2) + np.power(v - Q / 2, 2))
        if D >= r0 and D <= r1:
          filter[u, v] = 0
        else:
          filter[u, v] = 1

    # filtering
    filtered_img = np.multiply(freq_domain, filter)

    # restoring and normalizing image
    restored_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    restored_img = restored_img.real
    restored_img = (restored_img-np.min(restored_img))*(255/(np.max(restored_img)-np.min(restored_img)))

    # comparing with ref
    print(RMSE(restored_img, ref_img))

def laplacian(input_img, ref_img):

    # generating the frequency domain
    freq_domain = np.fft.fftshift(np.fft.fft2(input_img))

    # generating the filter
    filter = freq_domain.copy()

    P, Q = freq_domain.shape
    for u in range(P):
      for v in range(Q):
        filter[u, v] = -4 * np.power(np.pi, 2) * (np.power(u - P / 2, 2) + np.power(v - Q / 2, 2))

    # filtering
    filtered_img = np.multiply(freq_domain, filter)

    # restoring and normalizing image
    restored_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    restored_img = restored_img.real
    restored_img = (restored_img-np.min(restored_img))*(255/(np.max(restored_img)-np.min(restored_img)))

    # comparing with ref
    print(RMSE(restored_img, ref_img))

def gaussian(input_img, ref_img, r, c):

    # generating the frequency domain
    freq_domain = np.fft.fftshift(np.fft.fft2(input_img))

    # generating the filter
    filter = freq_domain.copy()

    P, Q = freq_domain.shape
    for u in range(P):
      for v in range(Q):
        x = ((np.power(u - P / 2, 2) / (2 * np.power(r, 2))) + (np.power(v - Q / 2, 2) / (2 * np.power(c, 2))))
        filter[u, v] = np.exp(-x)

    # filtering
    filtered_img = np.multiply(freq_domain, filter)

    # restoring and normalizing image
    restored_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    restored_img = restored_img.real
    restored_img = (restored_img-np.min(restored_img))*(255/(np.max(restored_img)-np.min(restored_img)))

    # comparing with ref
    print(RMSE(restored_img, ref_img))

def main():

  # reading inputs
  input_img_name = str(input().rstrip())
  ref_img_name = str(input().rstrip())
  filter_index = int(input())

  # loading images
  input_img = imageio.imread(input_img_name)
  ref_img = imageio.imread(ref_img_name)

  if filter_index == 0:
    radius = int(input())
    low_pass(input_img, ref_img, radius)

  elif filter_index == 1:
    radius = int(input())
    high_pass(input_img, ref_img, radius)

  elif filter_index == 2:
    r0 = int(input())
    r1 = int(input())
    band_stop(input_img, ref_img, r1, r0)
    
  elif filter_index == 3:
    laplacian(input_img, ref_img)
    
  elif filter_index == 4:
    r = int(input())
    c = int(input())
    gaussian(input_img, ref_img, r, c)

if __name__ == '__main__':
  main()