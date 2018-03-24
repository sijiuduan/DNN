import cv2

img = cv2.imread("demos/alcatraz2.jpg")
img = cv2.resize(img,(136 * 3,76 * 3))
# cv2.imshow("original",img)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# ------------------------------------------------------------------------
#使用SIFT
sift = cv2.xfeatures2d.SIFT_create(10)
keypoints, descriptor = sift.detectAndCompute(gray,None)

cv2.drawKeypoints(image = img,
                  outImage = img,
                  keypoints = keypoints,
                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color = (51,163,236))
cv2.imshow("SIFT",img)

# ------------------------------------------------------------------------
#使用SURF
surf = cv2.xfeatures2d.SURF_create(10)
keypoints, descriptor = surf.detectAndCompute(gray,None)

cv2.drawKeypoints(image = img,
                  outImage = img,
                  keypoints = keypoints,
                  flags = 0,#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color = (51,163,236))
cv2.imshow("SURF",img)

# ------------------------------------------------------------------------
#使用ORB
orb = cv2.ORB_create(10)
keypoints, descriptor = orb.detectAndCompute(gray,None)

cv2.drawKeypoints(image = img,
                  outImage = img,
                  keypoints = keypoints,
                  flags = 0,#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color = (51,163,236))
cv2.imshow("ORB",img)

cv2.waitKey(0)
cv2.destroyAllWindows()