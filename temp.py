# python -m pip install --user opencv-contrib-python numpy scipy matplotlib ipython jupyter pandas sympy nose
import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import time
import sys

def get_dimension():
    # using cam 
    # videocapture=cv2.VideoCapture(0)
    videocapture=cv2.VideoCapture('rtsp://172.20.10.1')
        #Reference for computer vision From 
    #https://gist.github.com/benmarwick/2b250d8ef3dbe36f817fbe2bf14aaa55
    def safe_div(x,y): # so we don't crash so often
        if y==0: return 0
        return x/y

    def nothing(x): # for trackbar
        pass

    def rescale_frame(frame, percent=60):  # make the video windows a bit smaller
        width = int(frame.shape[1] * percent/100)
        height = int(frame.shape[0] * percent/100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if not videocapture.isOpened():
        print("can't open camera")
        exit()
    start_time=time.time()    
    windowName="Webcam Live video feed"

    cv2.namedWindow(windowName)


    cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
    cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
    cv2.createTrackbar("iterations", windowName, 1, 10, nothing)

    showLive=True
    while(showLive):
        
        ret, frame=videocapture.read()
        frame_resize = rescale_frame(frame)
        if not ret:
            print("cannot capture the frame")
            exit()
    
        thresh= cv2.getTrackbarPos("threshold", windowName) 
        ret,thresh1 = cv2.threshold(frame_resize,thresh,255,cv2.THRESH_BINARY) 
        
        kern=cv2.getTrackbarPos("kernel", windowName) 
        kernel = np.ones((kern,kern),np.uint8) # square image kernel used for erosion
        
        itera=cv2.getTrackbarPos("iterations", windowName) 
        dilation =   cv2.dilate(thresh1, kernel, iterations=itera)
        erosion = cv2.erode(dilation,kernel,iterations = itera) # refines all edges in the binary image

        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  
        closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
        
        contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find contours with simple approximation 

        closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(closing, contours, -1, (128,255,0), 1)
        
        # focus on only the largest outline by area
        areas = [] #list to hold all areas

        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)

        max_area = max(areas)
        max_area_index = areas.index(max_area)  # index of the list element with largest area

        cnt = contours[max_area_index - 1] 

        cv2.drawContours(closing, [cnt], 0, (0,0,255), 1)
        
        def midpoint(ptA, ptB): 
            return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        # compute the rotated bounding box of the contour
        orig = frame_resize.copy()
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
    
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
    
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 1)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 1)
        cv2.drawContours(orig, [cnt], 0, (0,0,255), 1)
        
        
        # calculate the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))/10
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))/10
        
        # calculate the size of the object
        pixelperM = 0.48 # more to do here to get actual measurements that have meaning in the real world
        dimA = dA / pixelperM
        dimB = dB / pixelperM
        
        #Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        # prints object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimB), (int(tltrX - 10), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimA), (int(trbrX - 20), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # compute the center of the contour
        M = cv2.moments(cnt)
        cX = int(safe_div(M["m10"],M["m00"]))
        cY = int(safe_div(M["m01"],M["m00"]))
    
        # draw the contour and center of the shape on the image
        cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(orig, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        cv2.imshow(windowName, orig)
        cv2.imshow('', closing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time=time.time()
        elapsed = end_time - start_time
        if elapsed > 50:
            break
        videocapture.release()
        showLive = False
    #oj = (20, 185) 
    #cv2.putText(orig,"Length of your baggage is " + format(str(round(dimA/2.54,2))) + " inches",oj,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.destroyAllWindows()
    dimA_inches = dimA
    dimB_inches = dimB
    print("Length of your baggage is " + format(str(dimB_inches)) + " inches.")
    print ("Breadth of your baggage is " +format(str(dimA_inches)) + " inches.")

    if dimA_inches + dimB_inches < 56+36:
        return "This bag is a perfect fit for a Carry-On. Dimensions are: {}X{}".format(dimA_inches,dimB_inches) 
    elif dimA_inches + dimB_inches < 130:
        return "This bag will be a Cheked Baggage. Dimensions are: {}X{}".format(dimA_inches,dimB_inches) 

    #if

      
 