# img = cv2.imread('cat.jpg',IMREAD_COLOR)

# cv2.imshow('cat',img)

# cat = img[0:500,0:1000]
# cv2.imshow('cat',cat)

# new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow('newimg',new_img)

# img = cv2.imread('dog.jpg')

# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
# reflect_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)   #对称复制
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,value=0)

# cv2.imshow('replicate',replicate)
# cv2.imshow('reflect',reflect)
# cv2.imshow('reflect_101',reflect_101)
# cv2.imshow('wrap',wrap)
# cv2.imshow('constant',constant)


# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.title('plt_display dog')
# plt.show()
#
# plt.imshow(img)
# plt.title('cv2_display dog')
# plt.show()


# cat = cv2.imread('cat.jpg')
# dog = cv2.imread('dog.jpg')

# plt.imshow(cv2.cvtColor(cv2.addWeighted(cat,0.4,dog,0.6,0),cv2.COLOR_BGR2RGB))
# plt.title('fusion')
# plt.show()


# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # 第一个子图
# plt.subplot(2, 1, 1)  # 2行1列，当前是第1个子图
# plt.plot(x, y1)
# plt.title('正弦函数')
#
# # 第二个子图
# plt.subplot(2, 1, 2)  # 仍然是2行1列，但现在是第2个子图
# plt.plot(x, y2)
# plt.title('余弦函数')


# image_data = np.random.rand(10, 10)
# fig, axs = plt.subplots(1, 2)
#
# # 第一个子图显示图像
# axs[0].imshow(image_data, cmap='gray')
# axs[0].set_title('图像数据')
#
# # 第二个子图绘制直方图
# axs[1].hist(image_data.flatten(), bins=20)
# axs[1].set_title('直方图')
#
# # 显示图表
# plt.tight_layout()  # 自动调整子图布局，避免重叠
# plt.show()


# noise = cv2.imread('noise.jpg')
#
# blur = cv2.blur(noise,(2,2))
#
# plt.figure(figsize=(15,5))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(noise,cv2.COLOR_BGR2RGB))
#
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
# plt.show()


# 方框滤波，带归一化跟均值滤波一样
# box = cv2.boxFilter(noise, -1, (3, 3), normalize=True)
#
# plt.figure(figsize=(15, 5))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(noise, cv2.COLOR_BGR2RGB))
#
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
# plt.show()


# cat = cv2.imread('cat.jpg')
# dog = cv2.imread('dog.jpg')
#
# h,w,c = cat.shape
#
# m = np.zeros((h,w),dtype=np.uint8)
#
# m[100:400,200:400] = 255
# m[100:500,100:200] = 255
# r = cv2.add(cat,dog,mask=m)
# cv2.imshow('m',m)
# cv2.imshow('r',r)

# 人脸识别
# lena = cv2.imread('lena.jpg')
# hsv = cv2.cvtColor(lena,cv2.COLOR_BGR2HSV)
#
# min_hsv = np.array([0,10,80])
# max_hsv = np.array([33,255,255])
#
# mask = cv2.inRange(hsv,min_hsv,max_hsv)
#
# face = cv2.bitwise_and(lena,lena,mask=mask)
#
# cv2.imshow('lena',lena)
# cv2.imshow('face',face)


# lena = cv2.imread('lena.jpg')
#
# w,h,c= lena.shape
#
# key = np.random.randint(0,255,size=[w,h,c],dtype=np.uint8)
#
# autherize = cv2.bitwise_xor(lena,key)
# unautherize = cv2.bitwise_xor(autherize,key)
#
#
# cv2.imshow('lena',lena)
# cv2.imshow('key',key)
# cv2.imshow('autherize',autherize)
# cv2.imshow('unautherize',unautherize)


# lena = cv2.imread('lena.jpg')
# # plt.imshow(cv2.cvtColor(lena,cv2.COLOR_BGR2RGB))
# # plt.show()
# w,h,c = lena.shape
# rio = lena[90:270,200:340]
# key = np.random.randint(0,256,[w,h,c],dtype=np.uint8)
# lenaxorkey = cv2.bitwise_xor(lena,key)
# secretFace = lenaxorkey[90:270,200:340]
# lena[90:270,200:340] = secretFace
# enface = lena
# cv2.imshow('enface',enface)
#
# enfacexorkey = cv2.bitwise_xor(enface,key)
# secretFace = enfacexorkey[90:270,200:340]
# enface[90:270,200:340] = secretFace
# deface = enface
# cv2.imshow('deface',deface)


# name = 'wolf.png'
# o = cv2.imread(name)
# cv2.imshow('original',o)
# gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
#
# ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv2.imshow('binary',binary)
#
# src,contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# src1,contours1, hierarchy1 = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# if contours:
#     x = cv2.drawContours(o, contours, 0, (0,0,255), 3)
#     m00 = cv2.moments(contours[0])['m00']
#     m01 = cv2.moments(contours[0])['m01']
#     m10 = cv2.moments(contours[0])['m10']
#
#     cx = int(m10 / m00) if m00 != 0 else 0
#     cy = int(m01 / m00) if m00 != 0 else 0
#
#     cv2.putText(o, "wolf", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
#     cv2.imshow('result', o)


# o = cv2.imread('dongwu.jpeg')
#
# cv2.imshow('original',o)
# gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
# ret , binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# img , contours , hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#
# n = len(contours)
# for i in range(n):
#     print('contours['+ str(i) +']的面积=',contourArea(contours[i]))
#     cv2.drawContours(o,contours,i,(0,0,255),3)
# cv2.imshow('result',o)


# 图像计数
# import cv2
#
# imgname = 'smallcell.jpeg'
# origin =  cv2.imread(imgname)
#
# gray = cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
# # cv2.imshow('gray',gray)
# ret,binary = cv2.threshold(gray,195,255,cv2.THRESH_BINARY_INV)
# # cv2.imshow('binary',binary)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#
# erosion = cv2.erode(binary,kernel,iterations = 3)
# # cv2.imshow('erosion',erosion)
# dilation = cv2.dilate(erosion,kernel,iterations = 2)
# # cv2.imshow('dilation',dilation)
# gaussian = cv2.GaussianBlur(dilation,(5,5),0)
# cv2.imshow('gaussian',gaussian)
# _,contours,_ = cv2.findContours(gaussian,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#
# contoursOK = []
#
# for i in contours:
#     if cv2.contourArea(i) > 5:
#         contoursOK.append(i)
#
# draw = cv2.drawContours(origin,contoursOK,-1,(0,0,255),2)
#
# for i, j in zip(contoursOK, range(len(contoursOK))):
#     cx = int(cv2.moments(i)['m10'] / cv2.moments(i)['m00']) if cv2.moments(i)['m00'] != 0 else 0
#     cy = int(cv2.moments(i)['m01'] / cv2.moments(i)['m00']) if cv2.moments(i)['m00'] != 0 else 0
#     cv2.putText(draw, str(j), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#
# cv2.imshow('draw',draw)
# cv2.imshow('gaussian',gaussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 开运算
# import cv2
# import numpy as np
#
# img = cv2.imread('dongwu.jpeg')
# k = np.ones((10,10),np.uint8)
# r = cv2.morphologyEx(img,cv2.MORPH_OPEN,k)
# cv2.imshow('morphologyEX',r)
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('half_circular.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow('threshold',binary)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
opening1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
# cv2.imshow('opening1',opening1)

# dist_transform = cv2.distanceTransform(opening1,cv2.DIST_L2,3)
# ret , fore = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# cv2.imshow('fore',fore)
#
kernel = np.ones((2, 2), np.uint8)
opening2 = cv2.morphologyEx(opening1, cv2.MORPH_OPEN, kernel)
# cv2.imshow('opening2',opening2)

gaussianblur = cv2.GaussianBlur(opening2, (3, 3), 3)
# cv2.imshow("gaussianblur",gaussianblur)

gaussianblur = np.array(gaussianblur, np.uint8)
_, contours, hierarchy = cv2.findContours(gaussianblur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

count = 0
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    circle_img = cv2.circle(gaussianblur, center, radius, (0, 0, 0), 2)
    cv2.imshow("circle_img", circle_img)
    area = cv2.contourArea(cnt)
    area_circle = 3.14 * radius * radius
    if area / area_circle >= 0.7:
        img = cv2.putText(img, 'OK', center, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    else:
        img = cv2.putText(img, 'BAD', center, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    count += 1
img = cv2.putText(img, ('sum=' + str(count)), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
cv2.imshow('result', img)

cv2.waitKey()
cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
#
# img = cv2.imread("doggray.png")
# cv2.imshow("prototype",img)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret ,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# img, contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# (x,y), radius = cv2.minEnclosingCircle(contours[0])
# center = (int(x),int(y))
# radius = int(radius)
# cv2.circle(img,center,radius,(0,0,0),2,2)
#
# cv2.imshow("minicircle",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
