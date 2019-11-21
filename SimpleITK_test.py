import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

#  读取.nii格式文件并显示
path="./Leslie.jpg"
seg=sitk.ReadImage(path)
print("size: ",seg.GetSize()) #(width, height, depth)
print("width: ",seg.GetWidth())
print("height: ",seg.GetHeight())
print("depth: ",seg.GetDepth())
print("pixel: ",seg.GetPixel((0,0)))
print("physical position: ",seg.TransformIndexToPhysicalPoint((0,0)))
#将图片转化为数组
img=sitk.GetArrayFromImage(seg)
print("size: ",img.shape) #(depth, height, width)
plt.imshow(img[:,:],cmap="gray")
plt.show()

path="Leslie.jpg"
#按照特定的格式读入图片
reader=sitk.ImageFileReader()
reader.SetImageIO("JPEGImageIO")
reader.SetFileName(path)
image=reader.Execute()
#写入生成新图片
writer=sitk.ImageFileWriter()
writer.SetFileName("img.jpg")
writer.Execute(image)


