import numpy as np
import cv2
def flip(frame):
    frame = cv2.flip(frame, 0)
    return frame

def onFrame(input_frame,count):
    w = 9
    h = 6

    input_frame = cv2.resize(input_frame, (136 * 3, 76 * 3))

    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        count+=1
        cv2.imwrite(("cam/%03d.png" % count), input_frame)

        # 将角点在图像上显示
        cv2.drawChessboardCorners(input_frame, (w,h), corners, ret)
    return input_frame,count

def runCamera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280 / 2)
    cap.set(4, 1024 / 2)
    cap.set(15, 0.1)

    count = 0

    while(cap.isOpened()):
        # 从摄像头读取一帧，ret是表明成功与否
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128.0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 72.0)
        ret, frame = cap.read()
        if ret:
            #处理得到的帧，这里将其翻转
            frame,count = onFrame(frame,count)
            cv2.imshow('frame',frame)
        else:
            break
        # 监听键盘，按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ##释放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    runCamera()