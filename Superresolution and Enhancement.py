'''****************************************************************************
|                       USP - Universidade de São Paulo                       |
|            ICMC - Instituto de Ciências Matemáticas e de Computação         |
*******************************************************************************
|                    Bacharelado em Ciências de Computação                    |
|                                 2024/1                                      |
|                   SCC0251 - Image Processing and Analysis                   |
|                                                                             |
|              Author: Victor Lucas de Almeida Fernandes (12675399)           |
*******************************************************************************
>               Assignment 1: enhancement and superresolution
****************************************************************************'''

import numpy as np
import imageio.v3 as imageio

# compare enhanced image with a reference one
def RMSE(img, ref):

  N = img.shape[0]

  aux = np.float32(0.0)
  for i in range(N):
    for j in range(N):
      aux += np.power(ref[i, j] - img[i, j], 2).astype(np.float32)

  return np.sqrt(aux / (N * N)).astype(np.float32)

# returns the cumulative histogram of an image with nColors
def cumulativeHistogram(img, nColors):

  histogram = np.zeros(nColors).astype(np.int32)

  histogram[0] = np.sum(img == 0)
  for i in range(1, nColors):
    histogram[i] = np.sum(img == i) + histogram[i - 1]

  return histogram

# equalise the histogram of an image
def histogramEqualisation(img, nColors, histogram):

  N, M = img.shape
  equalisedImg = np.zeros([N, M]).astype(np.uint8)

  for z in range(nColors):
    s = (((nColors - 1) / float(M * N)) * histogram[z])
    equalisedImg[np.where(img == z)] = s.astype(np.uint8)

  return equalisedImg

def superresolution(img0, img1, img2, img3):

  # creating the high resolution img, shape is double of the original
  newShape = tuple(np.array(img0.shape) * 2)
  highRes = np.zeros(newShape)

  for i in range(newShape[0]):
    for j in range(newShape[0]):

      # if i and j are even, assign img0[i/2, j/2] to highRes[i, j]
      if i % 2 == 0 and j % 2 == 0:
        highRes[i, j] = img0[int(i/2), int(j/2)]

      # if j is even and i not, assign img1[(i-1)/2, j/2] to highRes[i, j]
      elif i % 2 != 0 and j % 2 == 0:
        highRes[i, j] = img1[int((i-1)/2), int(j/2)]

      # if i is even and j not, assign img2[i/2, (j-1)/2)] to highRes[i, j]
      elif i % 2 == 0 and j % 2 != 0:
        highRes[i, j] = img2[int(i/2), int((j-1)/2)]

      # if both are odd, assign img3[i/2, j/2] to highRes[i, j]
      else:
        highRes[i, j] = img3[int(i/2), int(j/2)]

  return highRes

def single_img_ch(img0, img1, img2, img3):

  # creating the histogram of each low img
  h0 = cumulativeHistogram(img0, 256)
  h1 = cumulativeHistogram(img1, 256)
  h2 = cumulativeHistogram(img2, 256)
  h3 = cumulativeHistogram(img3, 256)

  # equalising each histogram
  eqimg0 = histogramEqualisation(img0, 256, h0)
  eqimg1 = histogramEqualisation(img1, 256, h1)
  eqimg2 = histogramEqualisation(img2, 256, h2)
  eqimg3 = histogramEqualisation(img3, 256, h3)

  # combining the imgs
  return superresolution(eqimg0, eqimg1, eqimg2, eqimg3)

def joint_ch(img0, img1, img2, img3):

  # joining all 4 low images into one and getting its histogram
  concImg = np.concatenate((img0, img1, img2, img3), axis=0)
  h = cumulativeHistogram(concImg, 256)
  h = h / 4

  # equalising each image with the obtained histogram
  eqimg0 = histogramEqualisation(img0, 256, h)
  eqimg1 = histogramEqualisation(img1, 256, h)
  eqimg2 = histogramEqualisation(img2, 256, h)
  eqimg3 = histogramEqualisation(img3, 256, h)

  # combining the imgs
  return superresolution(eqimg0, eqimg1, eqimg2, eqimg3)

# auxiliar function to the gamma correction
def func(img, param):
  return np.floor(255 * (np.power(img.astype(np.int32) / 255.0, 1 / param))).astype(np.uint8)

def gamma_correction(img0, img1, img2, img3, param):

  # applying the gamma correction to each low image
  newImg0 = func(img0, param)
  newImg1 = func(img1, param)
  newImg2 = func(img2, param)
  newImg3 = func(img3, param)

  # combining the imgs
  return superresolution(newImg0, newImg1, newImg2, newImg3)

def main():

  # reading inputs
  imglow_name = str(input().rstrip())
  imghigh_name = str(input().rstrip())
  method_id = int(input())
  method_param = float(input())

  # loading images
  imglow0 = imageio.imread(imglow_name + '0.png')
  imglow1 = imageio.imread(imglow_name + '1.png')
  imglow2 = imageio.imread(imglow_name + '2.png')
  imglow3 = imageio.imread(imglow_name + '3.png')
  imghigh = imageio.imread(imghigh_name)

  # applying the enhancement method
  if method_id == 0:
    res = superresolution(imglow0, imglow1, imglow2, imglow3)

  elif method_id == 1:
    res = single_img_ch(imglow0, imglow1, imglow2, imglow3)

  elif method_id == 2:
    res = joint_ch(imglow0, imglow1, imglow2, imglow3)

  elif method_id == 3:
    res = gamma_correction(imglow0, imglow1, imglow2, imglow3, method_param)

  print('{:.4f}'.format(RMSE(res, imghigh)))

if __name__ == '__main__':
  main()