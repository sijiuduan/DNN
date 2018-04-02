# import numpy as np
import cv2
import time
import numpy as np
import math
from jimg.jcv.cvFilter import Filter
from jnn.ea import Ea

class ORBKeyFrame():
    def __init__(self):
        cv = Ea()
        cv.orb = cv2.ORB_create(500)
        cv.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
        cv.diff = 0.35
        self.cv = cv
        self.status = 0
        self.p1p2 = Ea()

    def debug(self):
        np.set_printoptions(linewidth=1500, precision=2, threshold=32, edgeitems=3)
        # Ea.show(self.org)
        # Ea.show(self.cur)
        # Ea.show(self.rich)
        # print(self.org.cv.detect.dsps.shape)

        # Ea.show(self.match)
        print(self.p1p2.kps.shape)
        print(self.p1p2.kps)

    def rich2cv(self):
        rich_kps = []
        rich_dsps = []
        for v in self.rich.values():
            kp = v.kp
            for dsp in v.dsps:
                rich_kps.append(kp)
                rich_dsps.append(dsp)
        return rich_kps, np.array(rich_dsps)

    def update(self, filter_frame, draw_frame):
        cv_kps, cv_dsps = self.cv.orb.detectAndCompute(filter_frame, None)

        # KEYFRAME_INITIALIZE
        if self.status == 0:
            # 重新初始化 原始关键帧 org keyframe
            org = Ea()
            org.cv.detect.kps = cv_kps
            org.cv.detect.dsps = cv_dsps
            org.cv.detect.count = len(cv_kps)

            self.org = org
            self.cur = Ea()
            self.cur.clone(org)

            self.org.filter_frame = filter_frame
            self.org.draw_frame = draw_frame

            # init rich data.
            _kpt = lambda x: int(x[0])*1000 + int(x[1])
            rich = Ea()
            for kp,dsp in zip(cv_kps,cv_dsps):
                rich[_kpt(kp.pt)].kp = kp
                rich[_kpt(kp.pt)].dsps = [dsp]

            self.rich = rich

            self.status = 1

        # KEYFRAME_READY
        elif self.status == 1:
            self.cur.cv.detect.kps = cv_kps
            self.cur.cv.detect.dsps = cv_dsps

        # KEYFRAME_ORG_MATCHES
        elif self.status == 2:
            self.cur.cv.detect.kps = cv_kps
            self.cur.cv.detect.dsps = cv_dsps

            matches = self.cv.bf.knnMatch(queryDescriptors=self.org.cv.detect.dsps,
                                          trainDescriptors=self.cur.cv.detect.dsps,
                                          k=2)

            good = [m for (m, n) in matches if m.distance < self.cv.diff * n.distance]
            self.cur.cv.matches = matches
            self.cur.cv.good = good

        # KEYFRAME_RICH_MATCHES
        elif self.status == 3:
            self.cur.cv.detect.kps = cv_kps
            self.cur.cv.detect.dsps = cv_dsps

            rich_kps, rich_dsps = self.rich2cv()
            self.cur.cv.rich.kps = rich_kps
            self.cur.cv.rich.dsps = rich_dsps

            matches = self.cv.bf.knnMatch(queryDescriptors=self.cur.cv.rich.dsps,
                                          trainDescriptors=self.cur.cv.detect.dsps,
                                          k=2)

            good = [m for (m, n) in matches if m.distance < self.cv.diff * n.distance]
            self.cur.cv.matches = matches
            self.cur.cv.good = good

            _p1p2 = []

            for (m, n) in matches:
                if m.distance < self.cv.diff * n.distance and m.distance > self.cv.diff * 0.90 * n.distance:
                    pt = self.cur.cv.rich.kps[m.queryIdx].pt
                    pt2 = cv_kps[m.trainIdx].pt
                    dsp = self.cur.cv.detect.dsps[m.trainIdx]

                    _kpt = lambda x: int(x[0]) * 1000 + int(x[1])

                    _p1p2.append([pt[0],pt[1],pt2[0],pt2[1]])

                    if len(self.rich[_kpt(pt)].dsps) < 50:
                        self.rich[_kpt(pt)].dsps.append(dsp)
                    else:
                        # print("max 50")
                        self.rich[_kpt(pt)].dsps.pop(1)
                        self.rich[_kpt(pt)].dsps.append(dsp)

            self.p1p2.kps = np.array(_p1p2)
            self.p1p2.org_frame = self.org.draw_frame
            self.p1p2.cur_frame = draw_frame
            pass

    def draw_cur_cv_detect(self, draw_frame, c=(51, 163, 236)):
        cv2.drawKeypoints(draw_frame, self.cur.cv.detect.kps, draw_frame, color=c)

    def draw_org_cv_maches(self,draw_frame):
        cv2.drawKeypoints(draw_frame, self.cur.cv.detect.kps, draw_frame, color=(163, 236, 51))

        _ipt = lambda x: (int(x[0]),int(x[1]))
        for g in self.cur.cv.good:
            pt1 = self.org.cv.detect.kps[g.queryIdx].pt
            pt2 = self.cur.cv.detect.kps[g.trainIdx].pt
            cv2.line(draw_frame, _ipt(pt1), _ipt(pt2), color=(0, 0, 255))

        matches_img = cv2.drawMatches(self.org.draw_frame, self.org.cv.detect.kps, draw_frame, self.cur.cv.detect.kps, self.cur.cv.good[:], draw_frame, flags=2)
        cv2.imshow("matches_img", matches_img)

    def draw_rich_cv_maches(self,draw_frame):
        cv2.drawKeypoints(draw_frame, self.cur.cv.detect.kps, draw_frame, color=(163, 236, 51))

        _ipt = lambda x: (int(x[0]),int(x[1]))
        for g in self.cur.cv.good:
            pt1 = self.cur.cv.rich.kps[g.queryIdx].pt
            pt2 = self.cur.cv.detect.kps[g.trainIdx].pt
            cv2.line(draw_frame, _ipt(pt1), _ipt(pt2), color=(0, 0, 255),thickness=2)

        matches_img = cv2.drawMatches(self.org.draw_frame, self.cur.cv.rich.kps,
                                      draw_frame, self.cur.cv.detect.kps,
                                      self.cur.cv.good[:], draw_frame, flags=2)
        cv2.imshow("matches_img", matches_img)

class ORBTracking(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.keyframe = ORBKeyFrame()
        self.frame_id = 0
        print("b,r,e,s,p")

    def onFrame(self, filter_frame, draw_frame):
        self.keyframe.update(filter_frame,draw_frame)

        if self.keyframe.status==1:
            self.keyframe.draw_cur_cv_detect(draw_frame)
        elif self.keyframe.status==2:
            self.keyframe.draw_org_cv_maches(draw_frame)
        elif self.keyframe.status==3:
            self.keyframe.draw_rich_cv_maches(draw_frame)
        pass

    def onKey(self,key):
        if key & 0xFF == ord('b'):
            print('b')
            self.keyframe.status = 2 #KEYFRAME_ORG_MATCHES
        elif key & 0xFF == ord('r'):
            print('r')
            self.keyframe.status = 3 #KEYFRAME_ORG_MATCHES
        elif key & 0xFF == ord('e'):
            print('e')
            self.keyframe.status = 0
        elif key & 0xFF == ord('s'):
            # cv2.imwrite("org.jpg",self.keyframe.p1p2.org_frame)
            # cv2.imwrite("cur.jpg", self.keyframe.p1p2.cur_frame)
            # Ea.dump(self.keyframe.p1p2.kps,"p1p2.pkl",".")
            print(type(self.keyframe.p1p2.kps))


            print('s')
        elif key & 0xFF == ord('p'):
            self.keyframe.debug()

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    from jimg.jcv.cvVideo import Video
    cam = Camera()
    # cam = Video()
    cam.addFilter(ORBTracking())
    cam.start()
#
