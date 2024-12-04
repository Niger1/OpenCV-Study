import cv2

import matplotlib.pyplot as plt

# img = cv2.imread('cat.jpg',IMREAD_COLOR)

# cv2.imshow('cat',img)

# cat = img[0:500,0:1000]
# cv2.imshow('cat',cat)

# new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow('newimg',new_img)

img = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)

top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)  # 对称复制
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                              value=0)

# cv2.imshow('replicate',replicate)
# cv2.imshow('reflect',reflect)
# cv2.imshow('reflect_101',reflect_101)
# cv2.imshow('wrap',wrap)
# cv2.imshow('constant',constant)

plt.imshow(cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB))
plt.title('upper cat')

cv2.waitKey(0)
cv2.destroyAllWindows()
