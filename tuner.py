import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from datetime import datetime

cap = cv2.VideoCapture(0)
marker_length = 0.125


with open('test.ply', 'w') as file:
    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write('element vertex WPISZ\n')
    file.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')

cam_pos=[]

focal = 463.36172654
width, height = 640,480
mtx = np.array([[463.36172654, 0.00000000e+00, 321.97022644],
                [0.00000000e+00, 464.1197524, 239.66525775],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[ 1.43799762e-01, -4.31395968e-01, -2.65579956e-04, -4.44358664e-03,
   3.26025213e-01]])

###------------------ ARUCO TRACKER ---------------------------
detections = 0
while (True):
    ret, frame = cap.read()
    print(frame.shape)
    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
        # (rvec-tvec).any() # get rid of that nasty numpy value array error
        if rvec is None:
            continue
        for coor in range(rvec.shape[0]):

            for i in range(0, ids.size):
                # draw axis for the aruco markers
                print(tvec[i])
                cv2.putText(frame, "tvec" + "{0:.2f}".format(tvec[i][0][0]) + " " + "{0:.2f}".format(
                    tvec[i][0][1]) + " " + "{0:.2f}".format(tvec[i][0][2]), (0, (i+1)*32), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, "rvec" + "{0:.2f}".format(rvec[i][0][0]) + " " + "{0:.2f}".format(
                    rvec[i][0][1]) + " " + "{0:.2f}".format(rvec[i][0][2]), (0, (i + 2) * 32), font, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)
                # cv2.imwrite('raw/'+str(detections) + "good.png", frame)
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)
                # cv2.imwrite('axis/'+str(detections) + "with_axis.png", frame)
                detections += 1
        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ', '
            if ids[i][0] == 0:
                cam_pos.append((-0.6-tvec[i][0][0] , 1.32-tvec[i][0][1] , 3-tvec[i][0][2] ))
                print("ID 1 dodano")
            if ids[i][0] == 1:
                cam_pos.append((-0.6-tvec[i][0][0] , 1.32-tvec[i][0][1] , 3.6-tvec[i][0][2] ))
                print("ID 2 dodano")
            if ids[i][0] == 2:
                cam_pos.append((-0.6-tvec[i][0][0], 1.32- tvec[i][0][1] , 4.2-tvec[i][0][2]))
                print("ID 3 dodano")

        cv2.putText(frame, "Id: " + strg, (0, 420), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        xp5pix=(tvec[0][0][0]+0.05)*focal/tvec[0][0][2]
        xm5pix=(tvec[0][0][0]-0.05)/tvec[0][0][2]
        y10pix=(tvec[0][0][1]+0.1)*focal/tvec[0][0][2]

        cv2.line(frame,(int(xm5pix+width//2), int(y10pix+height//2)),(int(xp5pix+width//2), int(y10pix+height//2)),(255,0,255,3))
        #cv2.circle(frame,(int(xpix+width//2), int(y10pix+height//2)), 5, (255, 0, 255), 2)
        print(len(cam_pos))
        f= open("test.ply","a")
        with f:
            if len(cam_pos):
                asd="{} {} {} 0 255 255\n".format(cam_pos[len(cam_pos)-1][0],cam_pos[len(cam_pos)-1][1],cam_pos[len(cam_pos)-1][2])
                print("zapisuje")
                f.write(asd)
    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 440), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
