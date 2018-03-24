import cv2
import numpy as np
from jimg.jcv.cvFilter import Filter

class ChessFilter(Filter):
    def __init__(self, save_count=0):
        Filter.__init__(self)
        self.chess_size = (9,6)
        w = self.chess_size[0]
        h = self.chess_size[1]

        if save_count > 1000:
            save_count = 999
        self.save_count = save_count

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        self.objp = np.zeros((w * h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # 储存棋盘格角点的世界坐标和图像坐标对
        self.objpoints = []  # 在世界坐标系中的三维点
        self.imgpoints = []  # 在图像平面的二维点

    def onFrame(self,flt_frame,draw_frame):
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(flt_frame, self.chess_size, None)
        if ret == True:
            # 亚像素坐标 貌似现在没用了。
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(flt_frame, corners, (11, 11), (-1, -1), criteria)
            if self.save_count > 0:
                cv2.imwrite(("output/camera_%03d.png" % self.save_count), draw_frame)
                self.save_count -= 1

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

            elif self.save_count == 0:
                print("objpoints:",len(self.objpoints))
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, flt_frame.shape[::-1], None, None)
                """
                参考：http://blog.csdn.net/shenxiaolu1984/article/details/50165635
                      http://blog.csdn.net/Sunshine_in_Moon/article/details/45440411
                cameraMatrix - 3*3的摄像机内矩阵
                distCoeffs - 4*1（具体尺寸取决于flags）。对图像坐标系进行进一步扭曲。这两个参数是内参数，可以把摄像机坐标系转换成图像坐标系。
                rvecs - 每一个视图的旋转向量。vector<Mat>类型，每个vec为3*1，可以用Rodrigues函数转换为3*3的旋转矩阵。
                tvecs - 每一个视图的平移向量。vector<Mat>类型，每个vec为3*1。
                """
                # print(ret, mtx, dist, rvecs, tvecs)
                print(mtx)
                self.save_count -= 1
                return

            # 将角点在图像上显示
            cv2.drawChessboardCorners(draw_frame, self.chess_size, corners, ret)

# # 去畸变
# img2 = cv2.imread('calib/chess.png')
# h,  w = img2.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
# dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# # 根据前面ROI区域裁剪图片
# #x,y,w,h = roi
# #dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)
#
# # 反投影误差
# total_error = 0
# for i in xrange(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     total_error += error
# print("total error: ", total_error/len(objpoints))

if __name__ == '__main__':
    from jimg.jcv.cvCamera import Camera
    cam = Camera()
    cam.addFilter(ChessFilter(30))
    cam.start()

