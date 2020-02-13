import glob

import cv2
import numpy as np
import tqdm

chessboard_res = (9,6)
####---------------------- CALIBRATION ---------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard of size (7 x 6) is used
objp = np.zeros((chessboard_res[1]*chessboard_res[0],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_res[0],0:chessboard_res[1]].T.reshape(-1,2)
# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# iterating through all calibration images
# in the folder
images = glob.glob('calib_images/*.jpg')
for i, fname in tqdm.tqdm(enumerate(images)):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find the chess board (calibration pattern) corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_res,None)
    # if calibration pattern is found, add object points,
    # image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # Refine the corners of the detected corners
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_res, corners2,ret)
        cv2.imshow("foto"+str(i), img)
        cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("ret", ret)
print("mtx", mtx)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)