import cv2
from jimg.jcv.cvFilter import Filter
from jnn.ea import Ea
class ORB(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.orb = cv2.ORB_create()
        # self.orb.setNLevels(3)
        # self.orb.setFirstLevel(1)

        # help(self.orb)
        self.status = 0

    def debug(self):

        print(help(self.keypoints[0]))
        pts = []
        for pt in self.keypoints:
            ePt = Ea()
            ePt.angle = pt.angle
            ePt.class_id = pt.class_id
            ePt.octave = pt.octave
            ePt.pt = (pt.pt[0],pt.pt[1])
            ePt.response = pt.response
            ePt.size = pt.size
            pts.append(ePt)

        Ea.show(pts,500)
        pass

    def drawPoint(self,pt,draw_frame):
        # cv2.circle(draw_frame, (round(pt[0]), round(pt[1])), 5, (255, 0, 255), -1)
        cv2.circle(draw_frame, (round(pt[0]), round(pt[1])), 3, (255, 0, 255))

    def drawKeyPoints(self,keypoints,draw_frame):
        for kp in keypoints:
            self.drawPoint(kp.pt, draw_frame)

    def onFrame(self, filter_frame, draw_frame):
        if self.status == 0:
            keypoints, descriptor = self.orb.detectAndCompute(filter_frame, None)
            # cv2.drawKeypoints(image=draw_frame,
            #                   outImage=draw_frame,
            #                   keypoints=keypoints,
            #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #                   color=(51, 163, 236))

            self.drawKeyPoints(keypoints,draw_frame)

            self.descriptor = descriptor
            self.keypoints = keypoints

    def onKey(self,key):
        if key & 0xFF == ord('d'):
            print('d')
            self.debug()

        if key & 0xFF == ord('e'):
            print('e')
            self.status = 0

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    cam = Camera()
    cam.addFilter(ORB())
    cam.start()

