'''
注意：OpenCV图片加载时默认通道顺序是BGR，图片类型为numpy矩阵
'''
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

cv2.setUseOptimized(True)  # 使用优化，默认是开启的
t1 = cv2.getTickCount()  # 参考点
# 读取图片
img1 = cv2.imread("./Leslie1.jpg")  # 第二个参数为0表示以灰度模式读入图片
img2 = cv2.imread("./Leslie2.jpg")
'''
OpenCV 中的加法与 Numpy 的加法是有所不同的。OpenCV 的加法是一种饱和操作，
而 Numpy 的加法是一种模操作。即当两数相加超过可以表示的最大值时，OpenCV
会取最大值，而Numpy会取模最大值后的值。
'''
img = img1 + img2  # 图片的加法（Numpy版）
cv2.imshow("add_Numpy", img)
cv2.waitKey(0)
img = cv2.add(img1, img2)  # 图片的加法（OpenCV版）
cv2.imshow("add_OpenCV", img)
cv2.waitKey(0)
# 获取从参考点到当前所经过的时间，输出的单位是秒
t2 = cv2.getTickCount()  # 当前时钟数
time = (t2 - t1) / cv2.getTickFrequency()  # 时钟频率
print("time: ", time)
'''
图像混合的公式如下：
    h(x)=(1-alpha)*f(x) + alpha*g(x) + beta
可以看作是两张图片的加权和
'''
# 图像的混合，其中最后一个参数即上式中的beta
img = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

# 将图片由BGR转化为灰色，也可以 该为cv2.COLOR_BGR2HSV由BGR该为HSV
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 图像阈值处理，参数分别为图片、初始阈值、最大值和使用的算法类型
# 第二个返回值是阈值化后的图像
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
'''
按位操作：
cv2.bitwise_not()
cv2.bitwise_and()
cv2.bitwise_or()
cv2.bitwise_xor()
'''
mask_inv = cv2.bitwise_not(mask)  # 将灰度图片取反
cv2.imshow("bitwise_not", mask_inv)
cv2.waitKey(0)
#
# # 创建一个名为image的新窗口，cv2.WINDOW_NORMAL表示可以调整大小
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.destroyWindow("image")  # 关闭指定窗口
# # 用zone替换掉原图中的某一部分
# zone = img[290:470, 130:320]
# img[370:550, 360:550] = zone
# cv2.imshow("img", img)
# # 等待指定毫秒的时间，参数为0会无限等待键盘输入，可以使图片正常显示
# key = cv2.waitKey(0)
# if key == 27:  # 按ESC键关闭
#     cv2.destroyAllWindows()  # 关闭所有窗口
# elif key == ord("s"):
#     cv2.imwrite("new_img.png", img)  # 保存图片
#     cv2.destroyAllWindows()  # 关闭所有窗口
#
# size = img.size  # 图片像素个数
# b, g, r = cv2.split(img)  # 通道拆分，该操作较为耗时
# img2 = cv2.merge([r, g, b])  # 通道合并
# # 为图像添加边框，参数分别为原图像、边框上下左右的宽度、边框类型。[0,0,255]代表蓝色
# img2 = cv2.copyMakeBorder(img2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255])
# plt.imshow(img2)
# plt.show()
# img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]  # 交换R、B两个通道
# plt.imshow(img)
# plt.show()
#
# '''
# cap=cv2.VideoCapture(r"D:\迅雷下载\生活大爆炸\S05E20.mkv")
# #FourCC编码是一个4字节编码，用来确定视频的编码格式
# fourcc=cv2.VideoWriter_fourcc(*"MJPG")
# #参数为输出视频名、编码格式、帧率和帧大小
# out=cv2. VideoWriter("out.avi",fourcc,25.0,(640,480))
# while cap.isOpened():
#     ret,frame=cap.read() #逐帧读入视频
#     # print("color: ",cv2.isColor())
#     # print("frame's width: ",cap.get(3)) #获取当前帧的宽度
#     # print("frame's height: ",cap.get(4)) #高度
#     # cap.set(3, 320)  # 设置当前帧的宽度为320
#     # cap.set(4, 240)
#     if ret==True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将彩色转换为灰色
#         cv2.imshow("frame", gray)
#         frame=cv2.flip(frame,0) #沿x轴方向旋转
#         out.write(frame)
#         if cv2.waitKey(1) & 0xFF==ord("q"):
#             break
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# '''
# # 获取红色的HSV值
# red = np.uint8([[[0, 0, 255]]])
# hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
# print("hsv_red: ", hsv_red)
#
# img = cv2.imread("./Leslie2.jpg")
# '''
# cv2.resize()函数可实现图片的缩放，当给定缩放因子fx，fy时，目标图片的宽高
# 将是原图的fx，fy倍。也可以直接给定目标图像的宽高。最后一个参数为插值算法，
# 当缩小图片时，推荐使用 cv2.INTER_AREA；在放大图片时，推荐使用 v2.INTER_CUBIC
# 和 v2.INTER_LINEAR（常用）。
# '''
# res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# height, width, channel = img.shape
# res = cv2.resize(img, (math.floor(0.5 * width), math.floor(0.5 * height)), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("shrink", res)
# tx = 100  # 图片往右平移的像素数
# ty = 50  # 图片往下平移的像素数
# # 构建图片平移用的移动矩阵
# matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
# # 仿射变换，参数一般为原图片、移动矩阵和目标图片大小
# res = cv2.warpAffine(img, matrix, (width, height))  # 图片平移
# cv2.imshow("translation", res)
# '''
# 获取旋转图片时所用的旋转矩阵，第一个参数是旋转中心，第二个参数是
# 逆时针旋转的角度，第三个参数是缩放因子
# '''
# matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 1)
# res = cv2.warpAffine(img, matrix, (width, height))  # 图片旋转
# cv2.imshow("rotate", res)
# # points1和points2分别为原图和结果图中的三个仿射变换参考点
# points1 = np.array([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
# points2 = np.array([[10, 100], [200, 50], [100, 250]], dtype=np.float32)
# matrix = cv2.getAffineTransform(points1, points2)  # 形成仿射变换矩阵
# res = cv2.warpAffine(img, matrix, (width, height))  # 图片旋转
# cv2.imshow("affine", res)
# # 图片的透视变换，需要在原图和结果图上找四个对应的点，且任3个点不可共线
# points1 = np.float32([[50, 50], [50, 500], [500, 50], [500, 500]])
# points2 = np.float32([[0, 0], [100, width], [height, 0], [width + 100, height]])
# matrix = cv2.getPerspectiveTransform(points1, points2)  # 形成透视变换矩阵
# res = cv2.warpPerspective(img, matrix, (width, height))  # 透视变化
# cv2.imshow("perspective", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# '''
# 简单图像阈值：当像素值高于阈值时就给该像素赋予一个新值，反之赋予另一个值
# cv2.threshold()的参数分别是：原图像（必须是灰度图）、初始阈值、最大值，使
# 用的阈值方法
#
# 自适应阈值：不再使用同一个数作为阈值，而是根据图像上不同的区域计算相应的
# 阈值。cv2.adaptiveThreshold()的参数有：
# src：原图像
# maxval：阈值的最大值
# thresh_type：阈值的计算方法，有两种取值：
#         cv2.ADAPTIVE_THRESH_MEAN_C:阈值取相邻区域的平均值
#            cv2.ADAPTIVE_THRESH_GAUSSIAN_C：阈值取相邻区域的加权和，权重为一个高斯窗口
# block_size：用来计算阈值的区域的大小
# C：一个常数，阈值等于平均值或加权平均值减去这个常数
#
# Otsu二值化：对一副双峰图像（图像直方图中有两个峰）自动根据其直方图计算出一个阈值
# 所用的函数仍是cv2.threshold()，但是需要多加个参数flag=cv2.THRESHOLD_OTSU，并且要
# 把阈值设为0，则该函数返回的第一个值就是最优的阈值。若不用Otsu，则第一个返回值为
# 设置的阈值。
# '''
# img_raw = cv2.imread("./Leslie2.jpg", 0)
# # 进行Otsu二值化
# ret, thresh0 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print("Otsu's ret: ", ret)
# ret, thresh1 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img_raw, 127, 255, cv2.THRESH_TOZERO_INV)
# img = cv2.medianBlur(img_raw, 5)  # 中值滤波
# thresh6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 cv2.THRESH_BINARY, 11, 2)
# thresh7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY, 11, 2)
# titles = ["Oringal", "Otsu", "Binary", "Binary_inv", "Trunc", "Tozero", "Tozero_inv",
#           "Adaptive Mean", "Adaptive Gaussian"]
# images = [img_raw, thresh0, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# '''
# 对图像的卷积操作可以看作是对图像进行滤波，通过对图像进行低通滤波（LPF）可以去除噪音，
# 模糊图像；高通滤波可以找到图像的边缘。
#
# 图像模糊（图像平滑）主要有四种：平均、高斯模糊、中值模糊和双边滤波。这四种滤波方式所
# 使用的卷积核矩阵的大小必须是奇数。平均滤波使用的卷积核是一个全 1/n 的矩阵，其中n是矩阵
# 元素数。高斯模糊使用的卷积核是矩阵中心最大，往四周按三维高斯分布递减的矩阵，可有效去除
# 高斯噪音。中值模糊是使用卷积框对应像素的中值来代替中心像素的值，可以有效去除椒盐噪声。
# 双边滤波与高斯模糊类似，但是还考虑到了像素之间的相似度，可以有效去除噪声而不影响边界。
# 它同时使用空间高斯权重和灰度值相似性高斯权重，前者保证只有邻近区域的像素对中心点有影响，
# 而后者保证只有和中心像素灰度值接近的像素点才会被用作模糊运算。
# '''
# img = cv2.imread("./Leslie2.jpg", 0)
# kernel = np.ones((5, 5), np.float32) / 25
# blur0 = cv2.filter2D(img, -1, kernel)  # 卷积操作
# blur1 = cv2.blur(img, (5, 5))  # 平均滤波
# # (5,5)是卷积核的大小，0,1分别是在x、y轴方向的标准差，若只填一个标准差，则两个方向的相同
# blur2 = cv2.GaussianBlur(img, (5, 5), 0, 1)  # 高斯模糊
# blur3 = cv2.medianBlur(img, 5)  # 中值模糊
# # 9是卷积核大小，两个75分别是空间高斯函数标准差和灰度值相似性高斯函数标准差
# blur4 = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
# images = [img, blur0, blur1, blur2, blur3, blur4]
# titles = ["original", "conv", "blur", "Gaussian_blur", "median_blur", "bilateral_blur"]
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# '''
# 形态学转换包括以下几种：
#
# 腐蚀：如果卷积核对应的原图像的像素值都为1，则中心像素则保持原值，反之变为0。可以将
#     前景物品（白色）的边界腐蚀掉，使物体看起来更小。
# 膨胀：如果卷积核对应的原图像的像素中有一个为1，则中心像素就为1，反之为0。可以增加
#     图像中的白色区域。
# 开运算：先腐蚀再膨胀，用于去除噪声。
# 闭运算：先膨胀再腐蚀，用于填充前景物体中的小洞，或前景物体上的小黑点。
# 形态学梯度：膨胀和腐蚀得到的图像之差，看起来是前景物体的轮廓。
# 礼帽：原图像和开运算得到的图像之差。
# 黑帽：闭运算和原图像得到的图像之差。
# '''
# img = cv2.imread("./1.png", 0)
# kernel = np.ones((5, 5), np.uint8)
# '''
# 可以用cv2.getStructuringElement()函数来构建椭圆形、圆形的卷积核，当第二个参数为
# cv2.MORPH_RECT时表示矩形核，为cv2.MORTH_ELLIPSE时为椭圆核，为cv2.MORTH_CROSS时为
# 十字形核。
# '''
# # cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# erosion = cv2.erode(img, kernel, iterations=1)  # 腐蚀
# dilation = cv2.dilate(img, kernel, iterations=1)  # 膨胀
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # 梯度
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # 礼帽
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # 黑帽
# images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat, img]
# titles = ["img", "erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat", "img"]
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# '''
# OpenCV提供了三种梯度滤波器，或者说高通滤波器，即Sobel,Scharr和Laplacian，其中前
# 两者是求一阶或二阶导数，Scharr是对Sobel的优化，Laplacian是求二阶导数。
#
# Sobel算子是高斯平滑和微分操作的结合体，抗噪能力好。3*3的Scharr卷积核为：
#     x_order: [[-3,0,3],[-10,0,10],[-3,0,3]]
#     y_order：[[-3,10,-3],[0,0,0],[3,10,3]]
# Laplacian算子可以使用二阶导数的形式定义,拉普拉斯滤波器使用的卷积核为：
#     kernel：[[0,1,0],[1,-4,1],[0,1,0]]
# '''
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# '''
# cv2.CV_64F是输出图像的深度（数据类型），若改为-1则和原图保持一致，参数1，0表示
# 只在x方向求一阶导数，最大可求二阶导数。当ksize=-1时会使用3*3的Scharr滤波器，效果
# 比Sobel滤波器好。
# '''
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# images = [img, laplacian, sobelx, sobely]
# titles = ["img", "laplacian", "sobelx", "sobely"]
# for i in range(4):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
'''
Canny边缘检测包括以下几个步骤：
1、噪声去除：用5*5的高斯滤波器去除噪声
2、计算图像梯度：使用Sobel算子计算水平方向和竖直方向的一阶导数（图像梯度）（Gx和Gy），
    并据其找到边界的梯度和方向，公式如下：
        Edeg_Gradient(G)=sqrt(Gx^2 + Gy^2)
        Angle(theta)=tan^(-1)(Gx/Gy)
    梯度方向一般总是与边界垂直，梯度方向可分为四类：垂直、水平和两个对角线。
3、非极大值抑制：对每个像素进行检查，若该点梯度是周围具有相同梯度方向的点中最大的，则

4、滞后阈值：设定两个阈值，minVal和maxVal，当图像灰度梯度高于maxVal时则被认为是真的边界，
    而那些低于minVal的边界则会被舍弃，若介于两者之间，则如果它与真正的边界点相连则为边界
    点，反之不是。
'''
img = cv2.imread("./Leslie2.jpg", 0)
# 参数分别为原图像、minVal、maxVal、Sobel卷积核大小（默认值为3）和L2gradient，最后一个参数
# 若为True则用以上提到的公式计算，反之使用 Edge_Gradient(G)=|Gx^2|+|Gy^2|，默认为False
edges = cv2.Canny(img, 100, 200, L2gradient=True)
cv2.imshow("edges", edges)
cv2.waitKey(0)
#''''
# 图像金字塔是为了处理分辨率不同的图片而构建的分辨率大的图片在下，小的在上的金字塔状结构。
# 图像金字塔有两种：高斯金字塔和拉普拉斯金字塔，前者中顶部的一层图片中的像素是下面相邻图片
# 的5个像素的高斯加权平均值，操作一次一张M*N的图片就变为了(M/2)*(N/2)的图片，这被成为Octave。
# 拉普拉斯金字塔得到的图片看起来就像边界图，可以有高斯金字塔计算得来，公式如下：
#     Li=Gi - PyrUp(Gi+1)
# '''
# lower_reso=cv2.pyrDown(img) #图片尺寸变小，分辨率降低，只生成一张图片
# higher_reso=cv2.pyrUp(lower_reso) #图片尺寸变大，分辨率不变
# cv2.imshow("lower", lower_reso)
# cv2.waitKey(0)
# cv2.imshow("higher", higher_reso)
# cv2.waitKey(0)
# laplacian=img-higher_reso
# cv2.imshow("laplacian", laplacian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# '''
# 轮廓就是将连续的点连在一起的曲线，具有相同的颜色或灰度。
# '''
# img=cv2.imread("./rect.png")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh=cv2.threshold(gray,127,255,0)
# '''
# 在寻找轮廓前要进行阈值化处理或Canny边界检测，轮廓查找函数会修改原始图像，该函数可以
# 在黑色背景中找出白色的物体。cv2.findContours()的参数为原图像、轮廓检索模式和轮廓近似
# 方法。其中当轮廓近似方法为cv2.CHAIN_APPROX_NONE时会存储轮廓上所有点的坐标，当为
# cv2.CHAIN_APPROX_SIMPLE时则会把冗余点去掉，压缩存储。返回值分别是：图像、轮廓和轮廓的
# 层次结构。其中轮廓是一个列表，每个轮廓是一组包含对象边界点的坐标。
# '''
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# '''
# cv2.drawContours()可以用来绘制轮廓，它的参数为原图像、轮廓、轮廓索引、颜色和厚度。其
# 中轮廓索引为-1时则会绘制所有轮廓。
# '''
# img=cv2.drawContours(img,contours,-1,(0,255,0),3)
# cv2.imshow("coutours",img)
# cv2.waitKey(0)
# #img=cv2.drawContours(img,contours,3,(0,255,0),3) #绘制第4个轮廓
#
# contours,hierarchy=cv2.findContours(thresh,1,2)
# cnt=contours[0]
# M=cv2.moments(cnt) #图像的矩（一个字典），可以用来计算图像的质心和面积等
# print("M:\n",M)
# cx=int(M["m10"]/M["m00"]) #质心x轴坐标
# cy=int(M["m01"]/M["m00"]) #质心y轴坐标
# area=cv2.contourArea(cnt) #轮廓面积
# perimeter=cv2.arcLength(cnt,True) #轮廓周长，第二个参数用来指定轮廓是否闭合
# epsilon=0.1*perimeter
# '''
# 轮廓近似就是将轮廓近似到另一只由更少的点组成的轮廓形状。cv2.approxPolyDP()
# 的第二个参数是epsilon，它是从原始轮廓到近似轮廓的最大距离，是一个准确度参数
# '''
# approx=cv2.approxPolyDP(cnt,epsilon,True)
#
# '''
# cv2.convexHull()可以检测一个曲线是否具有凸性缺陷并纠正，凸性缺陷就是图像凹进去
# 的部分。该函数的参数为：
# points：输入的轮廓
# hull：输出，通常不需要
# clockwise：是否为顺时针方向
# returnPoints：为True时会返回凸包上点的坐标，反之会返回与凸包点对应的轮廓上的点
# '''
# hull=cv2.convexHull(cnt,returnPoints=False) #要查找凸缺陷时，第二个参数必须是False
# #找到凸缺陷，返回值是一个数组，每一行是[起点，终点，最远的点，到最远点的近似距离]
# defects=cv2.convexityDefects(cnt,hull)
# k=cv2.isContourConvex(cnt) #检测曲线是否为凸的，返回值为True或False
# '''
# 边界矩形分为直矩形和旋转矩形两种，其中前者的边是平行于x、y轴的，cv2.boundingRect()
# 返回值是矩形左上角的坐标和矩形的宽和高。后者考虑到了对象的旋转，可以得到面积最小的
# 边界矩形，cv2.minAreaRect()的返回值除了x、y、w和h，还有旋转的角度。
# '''
# x,y,w,h=cv2.boundingRect(cnt) #直矩形
# img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# rect=cv2.minAreaRect(cnt) #旋转矩形
# box=np.int0(cv2.boxPoints(rect))
# cv2.drawContours(img,[box],0,(0,0,255),2)
# (x,y),radius=cv2.minEnclosingCircle(cnt) #最小外接圆
# center=(int(x),int(y))
# radius=int(radius)
# img=cv2.circle(img,center,radius,(0,255,0),2)
# ellipse=cv2.fitEllipse(cnt) #最小外接椭圆
# img=cv2.ellipse(img,ellipse,(0,255,255),2)
# cv2.imshow("rect",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# '''
# 长宽比：边界矩形的宽高比
# Extent：轮廓面积与边界矩形面积之比
# Solidity：轮廓面积与凸包面积之比
# Equivalent Diameter：与轮廓面积相等的圆的直径
# 方向：对象的方向，可用以下函数返回长轴和短轴的长度
#     (x,y),(MA,ma),angle=cv2.fitEllipse(cnt)
# '''
# mask=np.zeros(gray.shape,np.uint8) #掩模和像素点（获取构成对象的所有像素点）
# cv2.drawContours(mask,[cnt],0,255,-1) #第三个参数必须为-1，绘制填充的轮廓
# pixelpoints=np.transpose(np.nonzero(mask))
# #最小、最大值及它们的位置
# min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(gray,mask=mask)
# mean_val=cv2.mean(img,mask=mask) #平均颜色及平均灰度
# #极点，一个对象最上下左右的点
# topmost=tuple(cnt[cnt[:,:,1].argmin()][0])
# bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
# leftmost=tuple(cnt[cnt[:,:,0].argmin()][0])
# rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
# #求点(50,50)到轮廓的距离，若在轮廓内部则为正数，在轮廓上为0，反之为负数
# # 第三个参数如果为True则会计算最短距离，反之只会判断位置关系，此时返回值
# # 为 1，-1，0
# dist=cv2.pointPolygonTest(cnt,(50,50),True)
# '''
# cv2.matchShape()可以帮我们比较两个形状或轮廓的相似度，返回值越小则匹配越好。
# 即使图像发生了旋转，对匹配的影响也不会很大。它是根据Hu矩来计算的，Hu矩是归
# 一化中心距的线性组合，它对某些如缩放、旋转和镜像映射等变化具有不变形。
# '''
# # ret=cv2.matchShapes(cnt1,cnt2,1,0.0)
#
# '''
# 我们称外部的轮廓为父，内部的为子。每一个轮廓的信息包括[next,previous,first_child,
# parent]，next为同一级组织结构中的下一轮廓，previous为同一级结构中的前一轮廓，
# first_child为第一个子轮廓，parent为父轮廓。如果没有父或子，则为-1。
#
# cv2.findContours()函数中轮廓检索模式有以下几种：
# cv2.RETR_LIST：只提取所有轮廓，而不创建任何父子关系
# cv2.RETR_TREE：返回所有轮廓，并创建一个完整的组织结构列表
# cv2.RETR_CCOMP：返回所有轮廓，并将其分为两级组织结构，一个对象的外轮廓为第1级，而
#     对象内部空洞的轮廓为第2级
# cv2.RETR_EXTERNAL：只会返回最外边的轮廓（可能有多个），所有的子轮廓会被忽略
# '''
#
# '''
# 通过直方图可以看出图像的灰度分布情况，直方图的x轴是灰度值（0~255），y轴是图像中具有
# 同一灰度值的点的数目。
#
# 术语：
# BINS（histSize）：将256 个灰度值平均分为n份，每一份为一个bins
# DIMS：收集数据的参数数目
# RANGE：要统计的灰度值范围
#
# cv2.calcHist()统计一副图像的直方图，其参数为：
# images：原图像，图像格式为uint8或float32，需要用[]括起来
# channels：通道，如果是灰度图，则为[0]，若为彩色图则为[0],[1],[2]，分别对应BGR
# mask：掩模图像，没有设为None
# histSize：BIN的数目，需要用[]括起来
# ranges：像素范围，通常为[0,256]
# 以上参数中只有mask不需要加中括号
# '''
# #绘制灰色图像的直方图
# img=cv2.imread("./test.jpg",0)
# cv2.imshow("zhujiajian",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# hist=cv2.calcHist([img],[0],None,[256],[0,256]) #统计图像的直方图，hist为一维数组
# plt.hist(img.ravel(),256,[0,256])
# plt.title("gray")
# plt.show()
# #绘制彩色图像BGR三通道的直方图
# img=cv2.imread("./test.jpg")
# color=('b','g','r')
# for i,col in enumerate(color):
#     histr=cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     plt.xlim([0,256])
# plt.title("BGR")
# plt.show()
# #直方图均衡化，可改善图像的对比度，当图像的灰度几种在某一范围内，可以起到很好的改善效果
# img=cv2.imread("./gray.png",0)
# equ=cv2.equalizeHist(img) #直方图均衡化，输入是一张灰度图片
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.imshow("balance",equ)
# cv2.waitKey(0)
# #使用自适应的直方图均衡化，此时图像被分成许多小块（成为tiles），对每个小块分别进行
# # 直方图均衡化，这样做不易丢失细节
# clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# cl=clahe.apply(img)
# cv2.imshow("auto-balance",cl)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# '''
# 一维直方图仅考虑到了灰度值，对于彩色图像，其2D直方图还应该考虑到颜色（Hue）和饱和度
# （Saturation）
#
# 直方图反向投影可以用于图像分割或寻找ROI，它会输出与待搜索图像相同大小的图像，其中像素
# 值越高（越白）的点就越可能代表要搜索的目标。
# 在用OpenCV对图像做反向投影时，要先对直方图进行归一化处理，得一概率图像，然后用一个圆盘
# 形卷积核对其进行卷积操作，最后使用阈值进行二值化处理。
#
# '''
# img=cv2.imread("./target.jpg")
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# target=cv2.imread("./project.jpg")
# hsvt=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# hist=cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256]) #直方图
# #归一化处理，参数：原图像、输出图像、映射到结果图像中的最小值和最大值、归一化类型
# #cv2.NORM_MINMAX会将数组所有值映射到最小、最大值之间
# cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
# dst=cv2.calcBackProject([hsvt],[0,1],hist,[0,180,0,256],1) #反向映射
# disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #设置卷积核的形状和尺寸
# dst=cv2.filter2D(dst,-1,disc)
# ret,thresh=cv2.threshold(dst,50,255,0)
# thresh=cv2.merge((thresh,thresh,thresh)) #合并通道
# res=cv2.bitwise_and(target,thresh)
# res=np.vstack((target,thresh,res))
# cv2.imshow("project",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# '''
# 傅里叶变换常用来分析不同滤波器的频率特性，可以用2D离散傅里叶变换（DFT）来分析图像
# 的频域特性，实现DFT的一个快速算法被称为快速傅里叶变换（FFT）。
# 可把图像想象成是沿着两个方向采集的信号，所以对图像同时进行x和y轴方向的傅里叶变换，
# 就可以得到图像的频域表示（频谱图）。因图像中边界点和噪声处幅度变化比较大，所以称其
# 为图像中的高频分量，反之为低频分量。
# '''
# img=cv2.imread("./Leslie2.jpg",0)
# # cv2.dft()和cv2.idft()输出结果是双通道的，分别是结果的实数和虚数部分，输入图像应该
# # 是np.float32类型的
# dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT) #离散傅里叶变换
# dft_shift=np.fft.fftshift(dft) #将结果图像沿两个方向平移N/2
# #构建振幅图
# magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
# rows,cols=img.shape
# crow,ccol=math.floor(rows/2),math.floor(cols/2)
# '''
# 当图像的大小为2的幂次，或者2、3、5的倍数时，DFT的效率会变高，可以通过cv2.getOptimalDFTSize()
# 来获取最优的行数和列数，并对原图像补0.
# '''
# nrows=cv2.getOptimalDFTSize(rows)
# ncols=cv2.getOptimalDFTSize(cols)
# new_img=np.zeros((nrows,ncols))
# new_img[:rows,:cols]=img
# img=new_img
# # 掩模与低频区域对应的部分设为1，反之为0
# mask=np.zeros((rows,cols,2),np.uint8)
# mask[crow-30:crow+30,ccol-30:ccol+30]=1
# fshift=dft_shift*mask
# f_ishift=np.fft.ifftshift(fshift)
# img_back=cv2.idft(f_ishift)
# img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#
# images=[img,magnitude_spectrum,img_back]
# titles=["input image","magnitude spectrum1","magnitude spectrum2"]
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(images[i],cmap="gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# '''
# 模板匹配是在输入图像（大图）中找到模板图像（小图）位置的方法。
# '''
# img=cv2.imread("./maliao.jpg",0) #大图
# person=cv2.imread("./person.jpg",0) #模板图-人物
# money=cv2.imread("./money.jpg",0) #模板图-钱币
# w,h=money.shape[::-1]
# res=cv2.matchTemplate(img,money,cv2.TM_CCOEFF_NORMED)
# threshold=0.7
# #把res矩阵中的每一个像素值都和阈值比较，返回一个索引元组
# loc=np.where(res>=threshold)
# print("loc: ",loc)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
# cv2.imshow("res.png",img)
# cv2.waitKey(0)
#
# w,h=person.shape[::-1]
# methods=["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR",
#          "cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]
# for meth in methods:
#     temp=img.copy()
#     method=eval(meth)
#     #返回的是一个灰度图像，每个像素值表示该区域与模板的匹配程度
#     res=cv2.matchTemplate(temp,person,method) #模板匹配
#     #返回匹配程度最大的区域的左上角点坐标以及矩形的宽高
#     min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
#     #比较方法为以下两种时，最小值对应的位置才是匹配的区域
#     if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
#         top_left=min_loc
#     else:
#         top_left=max_loc
#     bottom_right=(top_left[0]+w,top_left[1]+h)
#     cv2.rectangle(img,top_left,bottom_right,255,2)
#     plt.subplot(121)
#     plt.imshow(res,cmap="gray")
#     plt.title("Matching Reult")
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(122)
#     plt.imshow(img,cmap="gray")
#     plt.title("Detected Point")
#     plt.xticks([])
#     plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

'''
霍夫（Hough）直线变换可以检测出可用数学表达式描述的形状，即使形状有一点破坏或扭曲
也可以使用该方法。对于直线来说，其表达式可以写为 y=ax+b，或 p=x*cos(theta)+y*sin(theta)
，其中p是原点到直线的垂直距离，theta是直线的垂线与x轴顺时针方向的夹角。在OpenCV中
右为x轴正方向，下为y轴正方向，当直线位于原点下方时，p为正，反之为负，theta的取值
范围是[0,180]，水平线的角度为90，竖直线的角度为0.
霍夫变换会先将一个二维数组初始化为0，其中数组的列数表示角度的精度，如果要精确到1度，
则应该有180列；行数表示像素的精度，如果要达到一个像素的精度，则行数应该为图像对角
线的距离。由于直线可以由(p,theta)表示，所以可以每次取直线上的一个点，遍历所有theta
的取值，并求出p的值，将(p,theta)对应的数组位置加1。不断重复以上过程，则最大值对应的
(p,theta)的值则为直线的两个参数。
'''
img=cv2.imread("./shudu.png")
img2=img.copy()
gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,apertureSize=3) #首先将图片转化为二值图片
#参数分别为：二值化图像、像素p的精确度、角度theta的精确度和阈值（检测到直线的最短长度）
# 当最后一个参数过大时，会导致lines为空
lines=cv2.HoughLines(edges,1,np.pi/180,130) #霍夫直线变换
for i in range(0,len(lines)):
    rho, theta=lines[i][0][0],lines[i][0][1]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow("houghLines",img2)
cv2.waitKey(0)
#
# minLineLength=80
# maxLineGap=10
# #cv2.HoughLinesP()是对cv2.HoughLines()的优化，它增加了线的最短长度和两条线之间的最大间隔
# # 这两个参数。返回值是线段起点和终点的坐标，该函数的阈值应该设的稍微小一点
# lines=cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
# for i in range(0,len(lines)):
#     x1,y1,x2,y2=lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3]
#     #参数：图像、起点、终点、颜色和画笔宽度
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow("houghLinesp",img)
# cv2.waitKey(0)
#
# '''
# 圆一共有三个参数：圆心坐标和半径，由于三维的累加器速度比较慢，所以OpenCV采取了
# 霍夫梯度法使用边界的梯度信息来求寻找图像中的圆形。
# '''
# img=cv2.imread("./opencv.jpg",0)
# img=cv2.medianBlur(img,5) #中值滤波
# cimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# '''
# 霍夫圆环检测函数cv2.HoughCircles()的参数为：
# image:输入的单通道图像
# method:检测圆的方法
# dp:图像分辨率和累加器分辨率之比，dp越大累加器数组越小
# minDist:圆心之间的最小距离
# param1:用于处理边缘检测的梯度值方法
# param2:cv2.HOUGH_GRADIENT方法的累加器阈值，阈值越小，检测到的圈子越多
# minRadius:最小半径
# maxRadius:最大半径
# '''
# circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,
#                          param2=45,minRadius=0,maxRadius=0)
# circles=np.uint16(np.around(circles))
# for i in circles[0,:]:
#     #参数：图像、圆心、半径、颜色和画笔宽度
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     cv2.circle(cimg, (i[0], i[1]), 2, (0,0,255), 3)
# cv2.imshow("detected circles",cimg)
# cv2.waitKey(0)
#
# '''
# 分水岭算法
# '''
# img=cv2.imread("./yingbi.jpg")
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #将图像转化为黑白图像，thresh是结果图
# ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("Otsu",thresh)
# cv2.waitKey(0)
#
# kernel=np.ones((3,3),np.uint8)
# # cv2.morphologyEx()是对图像进行形态学运算，参数为：原图像、操作类型、卷积核和操作
# # 被递归执行的次数。cv2.MORPH_OPEN是开运算，即先腐蚀后膨胀。可消除噪点
# opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
# sure_bg=cv2.dilate(opening,kernel,iterations=2) #2次膨胀
# cv2.imshow("sure_bg",sure_bg)
# cv2.waitKey(0)
#
# #获取非零像素到最近零像素点的最短距离
# dist_transform=cv2.distanceTransform(opening,1,5)
# #获取前景色，前景区域就是种子，从这里开始灌水
# ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# cv2.imshow("sure_fg",sure_fg)
# cv2.waitKey(0)
# #保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
# sure_fg=np.uint8(sure_fg)
# unknown=cv2.subtract(sure_bg,sure_fg)
# cv2.imshow("unknown",unknown)
# cv2.waitKey(0)
#
# #ret是连通与处理的边缘条数，也就是确定的区域数（种子数），markers是一个标签
# ret,markers=cv2.connectedComponents(sure_fg) #求图中的连通图
# #因分水岭算法需要将栅栏区域设置为0，而markers中背影区域原为0，故需要设为其他整数，可整体+1
# markers=markers+1
# markers[unknown==255]=0
# #根据种子开始灌水，算法会将找到的栅栏在markers中设为-1
# markers=cv2.watershed(img,markers=markers)
# img[markers==-1]=[0,255,0] #将栅栏区域设为绿色
# cv2.imshow("img",img)
# cv2.waitKey(0)

'''
GrabCut利用少量的人机交互来提取图像中的前景。首先用户要用一个矩形将前景完全框住，
然后算法会迭代分割，每次可以在分割不正确的地方画一笔（点击鼠标）则下一次就会得到
更好的分割结果。
该算法会根据用户画的矩形，使用一个高斯混合模型（GMM）对前景和背景建模，对于分类
未知的像素点，可以按照它们与已知分类的相似性来进行分类。这样就会根据像素的分布创建
一幅图，图中的节点就是像素点，除了像素点还有两个节点：Source_node和Sink_node，其中
所有前景像素和Source_node相连，背景像素和Sink_node相连。像素点与这两点的边权为与
它们属于同一类的概率。两个像素点之间的权重是它们的相似性。然后使用mincut算法对上图
进行分割，它会根据最低成本方程将图分为Source_node和Sink_node，成本方程就是被剪掉的
所有变的权重之和。重复该过程直至分类收敛。
'''
'''
img=cv2.imread("./ma.jpg")
mask=np.zeros(img.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,140,400,500)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype("uint8")
img=img*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()
'''

'''
角点向任何方向移动变化都会很大。Harris角点检测等检测方式具有旋转不变性，但没有缩放不变性
'''
# img=cv2.imread("./shudu.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray=np.float32(gray)
# '''
# 参数为：
# img:类型为float32的输入图像
# blockSize:角点检测中要考虑的邻域大小
# ksize:Sobel求导中使用的窗口大小
# k:Harris角点检测方程中的自由参数，取值范围为[0.04,0.06]
# '''
# dst=cv2.cornerHarris(gray,2,3,0.04)
# dst=cv2.dilate(dst,None)
# img[dst>0.01*dst.max()]=[0,255,0]
# cv2.imshow("dst",img)
#
# '''
# 亚像素级的角点检测，可以提供精度更大的角点检测，首先要找到Harris角点，然后将角点
# 的重心传给cv2.cornerSubPix()函数进行修正，还要为该函数定义一个迭代停止条件。
# '''
# img=cv2.imread("./shudu.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray=np.float32(gray)
# dst=cv2.cornerHarris(gray,2,3,0.04)
# dst=cv2.dilate(dst,None)
# ret,dst=cv2.threshold(dst,0.01*dst.max(),255,0)
# dst=np.uint8(dst)
# #寻找重心（centroids）
# ret,labels,stats,centroids=cv2.connectedComponentsWithStats(dst)
# #定义停止的标准
# criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.001)
# #重定义角点
# corners=cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# res=np.hstack((centroids,corners))
# res=np.int0(res) #可以省略小数点后的数字（非四舍五入）
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]]=[0,255,0]
# cv2.imshow("subpixel",img)
#
# #Shi-Tomasi角点检测算法在Harris算法的基础上改进了其角点检测的打分公式，适合在目标跟踪中使用
# img=cv2.imread("./shudu.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #参数：输入的灰度图像、要检测的最佳角点数目、角点的质量（介于0~1之间）和两个角点间的最短距离
# corners=cv2.goodFeaturesToTrack(gray,25,0.01,10)
# corners=np.int0(corners)
# for i in corners:
#     x,y=i.ravel() #将多维数组转化为一维数组
#     cv2.circle(img,(x,y),3,255,-1)
# plt.imshow(img)
# plt.show()

'''
SIFT（尺度不变特征变换）可以提取图像中的关键点并计算它们的描述符。
在使用该函数时必须用 pip install opencv-contrib-python==3.4.2.17来安装opencv-contrib版
安装之后在OpenCV显示图片时，应该只在最后一次显示的时候才加cv2.waitKey(0)
'''
img=cv2.imread("./test.jpg")
img2=img.copy()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp=sift.detect(gray,None) #找到图像中的关键点
cv2.drawKeypoints(gray,kp,img2) #绘制关键点，第三个参数为输出图像
cv2.imshow("keypoints",img2)
#不仅绘制代表关键点大小的圆圈而且还会绘制除关键点的方向
cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("keypoints2",img)
'''
SURF（加速稳健特征）算法是加速版的SIFT算法。
'''


cv2.waitKey(0)
print("end")

