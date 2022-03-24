from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def legacy_to_greyscale(image_path, out_dir):
  img = Image.open(image_path).convert('L')
  img.save(out_dir)

def to_greyscale(image_path, out_dir):
  img = cv.imread(image_path,0)
  _, b_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
  cv.imwrite(out_dir, b_img)

"""It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, h, w, n_row, n_col):
  plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
  plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
  for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
    plt.title(titles[i])
    plt.xticks(())
    plt.yticks(())

def pca(X, n_pc):
  mean = np.mean(X, axis=0)
  centered_data = X - mean
  _, _, V = np.linalg.svd(centered_data)
  components = V[:n_pc]
  return components, mean, centered_data

def reconstruction(Y, C, M, h, w, image_index):
  weights = np.dot(Y, C.T)
  centered_vector = np.dot(weights[image_index, :], C)
  recovered_image = (M + centered_vector).reshape(h, w)
  return recovered_image

def normalized(X):
  return X / np.linalg.norm(X)