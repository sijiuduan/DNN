# coding=utf-8
import cv2
from matplotlib import pyplot as plt

def imgAB():
    img1 = cv2.imread('alcatraz1.jpg')
    img2 = cv2.imread('alcatraz2.jpg')

    img1 = cv2.resize(img1, (300 * 3, 240 * 3))
    img2 = cv2.resize(img2, (300 * 3, 240 * 3))
    return img1, img2

def ptf2ptd(pt):
    return (int(pt[0]),int(pt[1]))

def ORB_match(img1,img2):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # knn筛选结果
    import time
    print(time.time())
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    # 查看最大匹配点数目
    print(time.time())
    # print(len(good), len(matches))

    # m = cv2.drawMatches(img1, kp1, img2, kp2, good[:10], img2, flags=2)
    # return m

    for g in good[:50]:
        # print( ptf2ptd(kp1[g.queryIdx].pt), ptf2ptd(kp2[g.trainIdx].pt))
        cv2.line(img1,ptf2ptd(kp1[g.queryIdx].pt), ptf2ptd(kp2[g.trainIdx].pt), color=(50,100,250),thickness=2)
        cv2.circle(img1, ptf2ptd(kp1[g.queryIdx].pt),5, color=(50, 100, 250))
    return img1

def BRISK_match(img1,img2):
    orb = cv2.BRISK_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # knn筛选结果
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    # 查看最大匹配点数目
    print(len(good), len(matches))

    m = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    return m

def SIFT_match(img1,img2):
    orb = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # knn筛选结果
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    # 查看最大匹配点数目
    print(len(good), len(matches))

    m = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    return m

def SURF_match(img1,img2):
    orb = cv2.xfeatures2d.SURF_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # knn筛选结果
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    # 查看最大匹配点数目
    print(len(good), len(matches))

    m = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    return m

def BRIEF_match(img1,img2):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp1 = star.detect(img1,None)
    kp2 = star.detect(img2,None)

    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)


    # knn筛选结果
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    # 查看最大匹配点数目
    print(len(good),len(matches))

    m = cv2.drawMatches(img1,kp1,img2,kp2,good,img2,flags=2)
    return m

def cvShow(img,title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pltShow(img,title=''):
    plt.imshow(img)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    imgA, imgB = imgAB()
    img = ORB_match(imgA, imgB)
    cvShow(img, "ORB_match")

