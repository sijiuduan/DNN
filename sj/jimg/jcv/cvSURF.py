import cv2
from jimg.jcv.cvFilter import Filter

class SURF(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.surf = cv2.xfeatures2d.SURF_create()

    def onFrame(self, filter_frame, draw_frame):
        keypoints, descriptor = self.surf.detectAndCompute(filter_frame, None)
        cv2.drawKeypoints(image=draw_frame,
                          outImage=draw_frame,
                          keypoints=keypoints,
                          flags=0,  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                          color=(0, 0, 236))


if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera

    cam = Camera()
    cam.addFilter(SURF())
    cam.start()

