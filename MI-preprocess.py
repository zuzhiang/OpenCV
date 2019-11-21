import cv2
import numpy as np
import pydicom
from PIL import Image
import SimpleITK as sitk
from skimage.io import  imread
import matplotlib.pyplot as plt

img=cv2.imread("./Leslie.jpg")
cv2.imshow("img",img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

filepath = './Leslie.jpg'
'''
PIL、OpenCV和skimage都可以对图片进行读入，其中前者读入的格式为
(width, height, ch)，经过np.asarray()处理后变为(height, width, ch)，
而后两者是(height, width, ch)，它们的返回值都是numpy的数组类型

除了opencv读入的彩色图片以BGR顺序存储外，其他所有图像库读入彩色图片都以RGB存储。
除了PIL读入的图片是img类之外，其他库读进来的图片都是以numpy 矩阵。
'''
cv2_im = cv2.imread(filepath)
print('cv2_im shape ',cv2_im.shape) # (height, width, ch)

im = Image.open(filepath)
print('PIL image size', im.size) # (width, height, ch)
pil_im = np.asarray(im)
print('pil_im shape ',pil_im.shape) # (height, width, ch)

sk_im = imread(filepath)
print('sk_im shape', sk_im.shape) # (height, width, ch)

#读取dicom文件并显示
print("pydicom")
path="./image_0"
img=pydicom.dcmread(path)
img=img.pixel_array
plt.imshow(img,cmap="gray")
plt.show()

#  读取.nii格式文件并显示
print("sitk")
path="./1.nii"
seg=sitk.ReadImage(path)
print("size: ",seg.GetSize()) #(width, height, depth)
print("width: ",seg.GetWidth())
print("height: ",seg.GetHeight())
print("depth: ",seg.GetDepth())
#将图片转化为数组
img=sitk.GetArrayFromImage(seg)
print("size: ",img.shape) #(depth, height, width)
for i in range(img.shape[0]):
    plt.imshow(img[i,:,:],cmap="gray")
    plt.show()