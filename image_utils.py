from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from scipy.signal import convolve2d
import numpy as np


def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array 

from scipy.signal import convolve2d
import numpy as np

def edge_detection(image_array):
   
    gray_image = np.mean(image_array, axis=2)

    kernelY = np.array([
       [1, 2, 1],
       [0, 0, 0],
       [-1, -2, -1]]) 
     
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1,0, 1] ])
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG

  
