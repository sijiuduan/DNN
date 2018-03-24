import cv2
from jimg.jcv.cvFilter import Filter

class ORB(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.orb = cv2.ORB_create()

    def onFrame(self, filter_frame, draw_frame):
        keypoints, descriptor = self.orb.detectAndCompute(filter_frame, None)
        cv2.drawKeypoints(image=draw_frame,
                          outImage=draw_frame,
                          keypoints=keypoints,
                          flags=0,  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                          color=(51, 163, 236))

        # cv2.line(draw_frame, (0,0), (100,100),color=(255,0,0))

        # self.old_frame = filter_frame

    def onKey(self,key):
        if key & 0xFF == ord('b'):
            print('b')

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    cam = Camera()
    cam.addFilter(ORB())
    cam.start()

