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
class Camera(Filter):
    def __init__(self):
        Filter.__init__(self)
        # self.name = "Camera"
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280 / 4)
        self.cap.set(4, 1024 / 4)
    def onFrame_capChess(self, frame):
        w = 9
        h = 6
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # count += 1
            # cv2.imwrite(("cam/%03d.png" % count), input_frame)

            # 将角点在图像上显示
            cv2.drawChessboardCorners(frame, (w, h), corners, ret)
        return frame

    def Rec(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                out.write(frame)
                cv2.imshow("Recording...", frame)

                # 监听键盘，按下q键退出
                keypress = cv2.waitKey(1)
                if keypress & 0xFF == ord('q'):
                    break
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

    def start(self):
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            # print(frame.shape)
            if ret == True:
                # gray = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.onFrame(gray, frame)
                cv2.imshow(self.getName(), frame)
            else:
                time.sleep(0.02)

            # 监听键盘，按下q键退出
            keypress = cv2.waitKey(1)
            if keypress & 0xFF == ord('q'):
                break
            else:
                self.onKey(keypress)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cam = Camera()
    # cam.addFilter(ChessFilter())
    # cam.Rec()
    # print(capPropId('FRAME_HEIGHT'))
    cam.start()