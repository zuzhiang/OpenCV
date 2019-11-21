#以下代码实现苹果和橘子图片的融合，但是有点问题，图片的像素数必须是2的幂次
import cv2 as cv
import numpy as np

apple=cv.imread("./apple.png")
orange=cv.imread("./orange.png")
gp_apple=[apple]
gp_orange=[orange]
for i in range(6):
    apple=cv.pyrDown(apple)
    gp_apple.append(apple)
    orange= cv.pyrDown(orange)
    gp_orange.append(orange)

lp_apple=[apple]
lp_orange=[orange]
for i in range(5,0,-1):
    apple=cv.pyrUp(gp_apple[i])
    apple=cv.subtract(gp_apple[i-1],apple)
    lp_apple.append(apple)
    orange=cv.pyrUp(gp_orange[i])
    orange= cv.subtract(gp_orange[i - 1], orange)
    lp_orange.append(orange)

LS=[]
for la,lb in zip(lp_apple,lp_orange):
    rows,cols,dpt=la.shape
    ls=np.hstack((la[:,:cols/2],lb[:,cols/2:]))
    LS.append(ls)
ls=LS[0]
for i in range(1,6):
    ls=cv.pyrUp(ls)
    ls=cv.add(ls,LS[i])
real=np.hstack((apple[:,:cols/2],orange[:,cols/2:]))
cv.imshow("pyramid_blending",ls)
cv.waitKey(0)
cv.imshow("direct_blending",real)
cv.waitKey(0)
cv.destroyAllWindows()