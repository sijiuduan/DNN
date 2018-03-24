# import numpy as np
import cv2
import time
import numpy as np
import math
from jimg.jcv.cvFilter import Filter
from jnn.ea import Ea


def _ipt(pt):
    return (int(pt[0]),int(pt[1]))

def _kpt(pt):
    return int(pt[0]) * 1000 + int(pt[1])

def _distance(pt1,pt2):
    rv =  math.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )
    return rv

class KeyFrame():
    def __init__(self,orb,bf,filter_frame,draw_frame):
        self.orb = orb
        self.bf = bf
        self.filter_frame = filter_frame
        self.draw_frame = draw_frame

        cv_kps, cv_dsps = self.orb.detectAndCompute(filter_frame, None)
        kps = []
        for kp in cv_kps:
            kps.append([kp.pt[0], kp.pt[1]])
        kps = np.array(kps)

        self.keyframe = Ea()
        self.keyframe.cv_kps = cv_kps
        self.keyframe.kps = kps
        self.keyframe.dsps = cv_dsps
        self.keyframe.img.filter = filter_frame
        self.keyframe.img.draw = draw_frame

        self.track = Ea()
        self.track.cv_kps = cv_kps
        self.track.kf.kps = []
        self.track.cr.kps = []
        self.track.cr.dsps = np.array([])
        self.track.img.filter = filter_frame
        self.track.img.draw = draw_frame

    def _match(self, query_kps, query_dsps, train_kps, train_dsps, _diff=0.35):
        rv = Ea()
        if len(query_kps)==0 or len(train_kps)==0:
            return rv

        matches = self.bf.knnMatch(queryDescriptors=query_dsps, trainDescriptors=train_dsps, k=2)
        good = [m for (m, n) in matches if m.distance < _diff * n.distance]

        if len(good) < 1:
            return rv

        rv.kf.kps = []
        rv.cr.kps = []
        rv.cr.dsps = []

        for g in good:
            kf_kp = query_kps[g.queryIdx]
            cr_kp = np.array([train_kps[g.trainIdx].pt[0],train_kps[g.trainIdx].pt[1]])
            cr_dsp = train_dsps[g.trainIdx]

            rv.kf.kps.append(kf_kp)
            rv.cr.kps.append(cr_kp)
            rv.cr.dsps.append(cr_dsp)

        rv.cr.dsps = np.array(rv.cr.dsps)
        # rv.good = good

        return rv

    @staticmethod
    def _parseMatch(cl: Ea, m: Ea):
        for kf_kp, cr_kp, cr_dsp in zip(m.kf.kps, m.cr.kps, m.cr.dsps):
            key = _kpt(kf_kp)
            cl[key].kf.kp = kf_kp
            cl[key].cr.kp = cr_kp
            cl[key].cr.dsp = cr_dsp

    @staticmethod
    def _parseCloud(cl: Ea):
        rv = Ea()
        rv.kf.kps = []
        rv.cr.kps = []
        rv.cr.dsps = []

        for v in cl.values():
            rv.kf.kps.append(v.kf.kp)
            rv.cr.kps.append(v.cr.kp)
            rv.cr.dsps.append(v.cr.dsp)
        rv.cr.dsps = np.array(rv.cr.dsps)
        return rv

    def update(self, filter_frame, draw_frame):
        self.draw_frame = draw_frame
        cr_kps, cr_dsps = self.orb.detectAndCompute(filter_frame, None)

        m1 = self._match(self.keyframe.kps, self.keyframe.dsps, cr_kps, cr_dsps)
        m2 = self._match(self.track.kf.kps, self.track.cr.dsps, cr_kps, cr_dsps)

        cl = Ea()
        KeyFrame._parseMatch(cl, self.track)
        KeyFrame._parseMatch(cl, m1)
        KeyFrame._parseMatch(cl, m2)

        self.track = KeyFrame._parseCloud(cl)
        self.track.img.filter = filter_frame
        self.track.img.draw = draw_frame
        self.track.cv_kps = cr_kps

        # for p1,p2 in zip(m2.kf.kps, m2.cr.kps):
        #     cv2.line(draw_frame, _ipt(p1), _ipt(p2), color=(0, 0, 255),thickness=2)
        #
        # for p1,p2 in zip(m1.kf.kps, m1.cr.kps):
        #     cv2.line(draw_frame, _ipt(p1), _ipt(p2), color=(0, 255, 255),thickness=1)

        return

    def update_bk(self, filter_frame, draw_frame):
        cr_kps, cr_dsps = self.orb.detectAndCompute(filter_frame, None)

        matches = self.bf.knnMatch(queryDescriptors=self.dsps[:self.orgKpCount], trainDescriptors=cr_dsps, k=2)
        good = [m for (m, n) in matches if m.distance < 0.35 * n.distance]


        m2 = self.bf.knnMatch(queryDescriptors=self.dsps[self.orgKpCount:-1], trainDescriptors=cr_dsps, k=2)
        g2 = [m for (m, n) in m2 if m.distance < 0.35 * n.distance]
        print("m1:", len(good), "m2:", len(g2))


        if len(good) < 1:
            return # Do nothing.

        match_kf_kps = []
        match_cr_kps = []
        match_cr_dsps = []

        for g in good[:-1]:
            match_kf_kp = self.kps[g.queryIdx]
            match_cr_kp = [cr_kps[g.trainIdx].pt[0],cr_kps[g.trainIdx].pt[1]]
            match_cr_dsp = cr_dsps[g.trainIdx]

            match_kf_kps.append(match_kf_kp)
            match_cr_kps.append(match_cr_kp)
            match_cr_dsps.append(match_cr_dsp)

        match_kf_kps = np.array(match_kf_kps)
        match_cr_kps = np.array(match_cr_kps)

        for p1, p2 in zip(match_kf_kps, match_cr_kps):
            cv2.line(draw_frame, _ipt(p1), _ipt(p2), color=(0, 0, 255))

        # 判断整体平移是否有一定距离：
        # mov = match_kf_kps - match_cr_kps
        # mov = mov * mov
        # mov = np.sum(mov, axis=1)
        # mov = np.sqrt(mov)
        #
        # mov = mov.mean()
        # print("Frame Move:", mov, "Match Count:", len(good))

        old_kps = self.kps[self.orgKpCount:-1]
        old_dsps = self.dsps[self.orgKpCount:-1]

        cl = Ea()
        for kp, dsp in zip(old_kps, old_dsps):
            key = _kpt(kp)
            cl[key].kp = kp
            cl[key].dsp = dsp

        for kp, dsp in zip(match_kf_kps, match_cr_dsps):
            key = _kpt(kp)
            cl[key].kp = kp
            cl[key].dsp = dsp

        out_kps = []
        out_dsps = []
        for v in cl.values():
            out_kps.append(v.kp)
            out_dsps.append(v.dsp)

        out_dsps = np.array(out_dsps)

        self.kps = np.r_[(self.kps)[:500] , out_kps]
        self.dsps = np.r_[(self.dsps)[:500], out_dsps]


class OrbMatch(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.orb = cv2.ORB_create(300)
        self.status = 0
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframe = Ea()

    def onFrame(self, filter_frame, draw_frame):
        if self.status==0:
            self.kf = KeyFrame(self.orb,self.bf,filter_frame,draw_frame)

            cv2.drawKeypoints(image=draw_frame,
                              outImage=draw_frame,
                              keypoints=self.kf.keyframe.cv_kps,
                              flags=0,
                              color=(51, 163, 236))

        elif self.status==1:
            self.kf.update(filter_frame,draw_frame)

            cv2.drawKeypoints(image=draw_frame,
                              outImage=draw_frame,
                              keypoints=self.kf.track.cv_kps,
                              flags=0,
                              color=(163, 236, 51))

            for p1,p2 in zip(self.kf.track.kf.kps, self.kf.track.cr.kps):
                cv2.line(draw_frame, _ipt(p1), _ipt(p2), color=(0, 0, 255),thickness=1)

            # self.kf.drawMatchPoints()

            # match_img = cv2.drawMatches(self.kf.keyframe.img.draw, self.kf.keyframe.cv_pts, draw_frame, self.kf.track.cv_pts, good[:], draw_frame, flags=2)
            # cv2.imshow("match", match_img)




    def onFrame2(self, filter_frame, draw_frame):
        if self.status == 0:
            self.keyframe[0].kps, self.keyframe[0].dsps = self.orb.detectAndCompute(filter_frame, None)
            self.keyframe.img = draw_frame

            cv2.drawKeypoints(image=draw_frame,
                              outImage=draw_frame,
                              keypoints=self.keyframe[0].kps,
                              flags=0,
                              color=(51, 163, 236))

        elif self.status == 1:
            kf_kps, kf_dsps = self.keyframe[0].kps, self.keyframe[0].dsps
            cr_kps, cr_dsps = self.orb.detectAndCompute(filter_frame, None)

            matches = self.bf.knnMatch(queryDescriptors=kf_dsps, trainDescriptors=cr_dsps, k=2)
            good = [m for (m, n) in matches if m.distance < 0.55 * n.distance]

            match_kf_kps = []
            match_cr_kps = []
            match_cr_dsps = []

            for g in good[:-1]:
                match_kf_kp = kf_kps[g.queryIdx]
                match_cr_kp = cr_kps[g.trainIdx]
                match_cr_dsp = cr_dsps[g.trainIdx]

                match_kf_kps.append(match_kf_kp)
                match_cr_kps.append(match_cr_kp)
                match_cr_dsps.append(match_cr_dsp)

                cv2.line(draw_frame, _ipt(match_kf_kp.pt), _ipt(match_cr_kp.pt), color=(0, 0, 255))

                # print(_distance(match_kf_kp.pt,match_cr_kp.pt))

            if len(match_cr_kps)>0:
                old_kps  = self.keyframe[0].kps[500:-1]
                old_dsps = self.keyframe[0].dsps[500:-1]

                cl = Ea()
                for kp,dsp in zip(old_kps,old_dsps):
                    key = _kpt(kp.pt)
                    cl[key].kp = kp
                    cl[key].dsp = dsp

                for kp,dsp in zip(match_kf_kps,match_cr_dsps):
                    key = _kpt(kp.pt)
                    cl[key].kp = kp
                    cl[key].dsp = dsp

                out_kps = []
                out_dsps = []
                for v in cl.values():
                    out_kps.append(v.kp)
                    out_dsps.append(v.dsp)

                out_dsps = np.array(out_dsps)

                # print(len(out_kps))

                self.keyframe[0].kps  = (self.keyframe[0].kps)[:500] + out_kps
                self.keyframe[0].dsps = np.r_[(self.keyframe[0].dsps)[:500], out_dsps]

            match_img = cv2.drawMatches(self.keyframe.img, kf_kps, draw_frame, cr_kps, good[:], draw_frame, flags=2)
            cv2.imshow("match", match_img)

            cv2.drawKeypoints(image=draw_frame,
                              outImage=draw_frame,
                              keypoints=cr_kps,
                              flags=0,
                              color=(163, 236, 51))


    def onKey(self,key):
        if key & 0xFF == ord('b'):
            print('b')
            self.status = 1
        elif key & 0xFF == ord('e'):
            print('e')
            self.status = 0
        elif key & 0xFF == ord('s'):
            print('s')

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    cam = Camera()
    cam.addFilter(OrbMatch())
    cam.start()

