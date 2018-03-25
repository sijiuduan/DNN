import numpy as np
import cv2
import time
from jimg.jcv.cvFilter import Filter

"""
camera matrix:
[[ 633.33470918    0.          301.33976143]
 [   0.          634.75307769  241.68611273]
 [   0.            0.            1.        ]]
"""
class Video(Filter):
    def __init__(self):
        Filter.__init__(self)
        self.cap = cv2.VideoCapture('output.avi')

    def start(self):
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.onFrame(gray, frame)
                cv2.imshow(self.getName(), frame)
            else:
                self.cap.retrieve()
                # print('ret==False')
                self.cap.release()
                self.cap = cv2.VideoCapture('output.avi')
                time.sleep(0.05)

            # 监听键盘，按下q键退出
            keypress = cv2.waitKey(1)
            if keypress & 0xFF == ord('q'):
                break
            else:
                self.onKey(keypress)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cam = Video()
    cam.start()