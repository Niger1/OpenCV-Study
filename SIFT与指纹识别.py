import cv2


#
# fp = cv2.imread('finger.jpeg')
# cv2.imshow('origin',fp)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(fp, None)
#
# cv2.drawKeypoints(fp, kp, fp)
#
# print('关键点的个数：',len(kp))
# print('前5个关键点；',kp[:5])
# print('第一个关键点的坐标：',kp[0].pt)
# print('第一个关键点的区域：',kp[0].size)
# print('第一个关键点的角度：',kp[0].angle)
# print('第一个关键点的响应：',kp[0].response)
# print('第一个关键点的层数：',kp[0].octave)
# print('第一个关键点的类id：',kp[0].class_id)
# print('第一个关键但的描述符：',des[0])
#
# cv2.imshow('visial_fp',fp)
#

def mysift_bfmatcher(a, b):
    sift = cv2.xfeatures2d.SIFT_create()
    kpa, desa = sift.detectAndCompute(a, None)
    kpb, desb = sift.detectAndCompute(b, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desa, desb, k=2)
    # good = [[m] for m, n in matches if m.distance < 0.8*n.distance]
    good = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good.append([m])

    result = cv2.drawMatchesKnn(a, kpa, b, kpb, good, None, flags=2)
    return result


def mysift_flannmatcher(a, b):
    sift = cv2.xfeatures2d.SIFT_create()
    kpa, desa = sift.detectAndCompute(a, None)
    kpb, desb = sift.detectAndCompute(b, None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(desa, desb, 2)
    good = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good.append([m])

    result = cv2.drawMatchesKnn(a, kpa, b, kpb, good, None, flags=2)
    return result


if __name__ == '__main__':
    a = cv2.imread('finger_a.jpeg')
    b = cv2.imread('finger_b.jpeg')
    c = cv2.imread('finger_all.jpg')
    d = cv2.imread('finger_leftdown.jpg')
    # m1 = mysift_bfmatcher(a, b)
    m2 = mysift_bfmatcher(c, d)
    # m3 = mysift_flannmatcher(a, b)
    m4 = mysift_flannmatcher(c, d)
    # cv2.imshow('a b with bfmatcher',m1)
    cv2.imshow('c d with bfmatcher', m2)
    # cv2.imshow('a b with flannmatcher',m3)
    cv2.imshow('c d with flannmatcher', m4)

cv2.waitKey()
cv2.destroyAllWindows()
