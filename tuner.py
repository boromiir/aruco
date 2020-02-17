import numpy as np
import cv2
import cv2.aruco as aruco
import glob

marker_size = 0.12
cap = cv2.VideoCapture(0)

def get_x_y_z(tvec, rotation, tvec_relative=[marker_size/2, marker_size/2, 0]):
    
    tvec_relative = np.expand_dims(np.array(tvec_relative), -1)
    
    print('tvec', tvec)
    print('rot', np.dot(rotation,tvec_relative) )
    x_y_z = tvec + np.squeeze(np.dot(rotation,tvec_relative))
    print('x_y_z is', x_y_z)
    return x_y_z

def raytracing(x,y,z, focal):
    assert z!=0
    x_on_screen = x/z*focal
    y_on_screen = y/z*focal
    return x_on_screen, y_on_screen

def drawer_point(frame, x, y):
    y_max,x_max, *non_used = frame.shape
    try:
        assert 0 < x < x_max
    except AssertionError as e:
        print(e, x, x_max)
    try:
        assert 0 < y < y_max
    except AssertionError as e:
        print(e, y, y_max)
    center_coordinates = (int(x), int(y)) 
  
    # Radius of circle 
    radius = 10
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 7
    
    # Using cv2.circle() method 
    # Draw a circle with blue line borders of thickness of 2 px 
    frame_circled = cv2.circle(frame, center_coordinates, radius, color, thickness, lineType=-1) 
    return frame_circled

def drawer_rect(frame, x1, y1, x2, y2):
    y_max,x_max, *non_used = frame.shape
    try:
        assert 0 < x < x_max
    except AssertionError as e:
        print(e, x, x_max)
    try:
        assert 0 < y < y_max
    except AssertionError as e:
        print(e, y, y_max)
    pt1 = (int(x1), int(y1)) 
    pt2 = (int(x2), int(y2))
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.circle() method 
    # Draw a circle with blue line borders of thickness of 2 px 
    frame_circled = cv2.rectangle(frame, pt1, pt2, color, thickness) 
    return frame_circled

focal = 650
width, height = 640,480
mtx = np.array([[  focal,   0.00000000e+00,   width//2],
       [  0.00000000e+00,   focal,   height//2],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

dist = np.array([[-0.12481632, -0.00360626,  0.00294025, -0.00138677,  0.52909508]])


###------------------ ARUCO TRACKER ---------------------------
while(1):
    ret, frame = cap.read()
    if frame is None: 
        continue
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
        print("znaleziono coś")
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error
        rotation = cv2.Rodrigues(rvec)[0]
        #print("rodriguez to", cv2.Rodrigues(rvec))
        for i in range(0, ids.size):
            # draw axis for the aruco markers
            cv2.imwrite("good.png", frame)
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)
            cv2.imwrite("with_axis.png", frame)
            #print(rotation)
            x,y,z = get_x_y_z(tvec[0,0], rotation, [0.01*1.5*12, -0.01*7.3*12, 0.01*0.5*12])
            x1,y1 = raytracing(x, y, z, focal)
            x,y,z = get_x_y_z(tvec[0,0], rotation, [0.01*2.9*12, -0.01*9.0*12, 0.01*0.5*12])
            x2,y2 = raytracing(x, y, z, focal)
            frame = drawer_rect(frame, x1+width/2,y1+height/2, x2+width/2,y2+height/2)
            
            #frame = drawer_point(frame, x2+width/2,y2+height/2)
        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)


        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame

    cv2.imshow('frame',frame)
    cv2.waitKey(10)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
