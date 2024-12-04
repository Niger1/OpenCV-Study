import cv2


def getplate(image):
    raw_image = image.copy()
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用Sobel函数进行边缘处理
    SobelX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(SobelX)
    image = absX
    # cv2.imshow('image',image)
    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('threshold',binary)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelY)
    # cv2.imshow('close', image)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelX)
    # cv2.imshow('open', image)

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    image = cv2.medianBlur(image, 15)
    # cv2.imshow('medianBlur',image)

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(raw_image.copy(), contours, -1, (0, 0, 255), 2)
    # cv2.imshow('contours',image)

    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if weight > (height * 3.5) and weight < (height * 6):
            plate = raw_image[y:y + height, x:x + weight]
    return plate


def preprocessor(image):
    image = cv2.GaussianBlur(image, (1, 1), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    # image = cv2.dilate(binary,kernel)
    # cv2.imshow('preprocessor',binary)
    return binary


def splitplate(image, origin):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        chars.append(rect)
        cv2.rectangle(origin, (x, y), (x + weight, y + height), (0, 0, 255), 1)
    cv2.imshow('contours2', origin)

    chars = sorted(chars, key=lambda s: s[0], reverse=False)

    platechars = []
    for rect in chars:
        if (rect[3] > (rect[2] * 1.5) and rect[3] < (rect[2] * 8)) and (rect[2] > 1):
            platechar = origin[y:y + height, x:x + weight]
            platechars.append(platechar)

    for i, img in enumerate(platechars):
        cv2.imshow('char' + str(i), img)


if __name__ == '__main__':
    image = cv2.imread("cp5.jpeg")
    pre_plate = getplate(image)
    cv2.imshow('plate', pre_plate)
    plate = preprocessor(pre_plate)
    splitplate(plate, pre_plate)

cv2.waitKey()
cv2.destroyAllWindows()
