import cv2
from jimg.jcv.cvFilter import Filter

class SIFT(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.sift = cv2.xfeatures2d.SIFT_create()
        # keypoints, descriptor = sift.detectAndCompute(gray, None)

    def onFrame(self, filter_frame, draw_frame):
        keypoints, descriptor = self.sift.detectAndCompute(filter_frame, None)
        cv2.drawKeypoints(image=draw_frame,
                          outImage=draw_frame,
                          keypoints=keypoints,
                          flags=0,  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                          color=(51, 163, 236))

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    cam = Camera()
    cam.addFilter(SIFT())
    cam.start()

