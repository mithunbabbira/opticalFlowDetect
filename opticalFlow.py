import numpy as np
import cv2






# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html


# py_lucas_kanade(method)

corner_track_params = dict(maxCorners = 10, qualityLevel = 0.3, minDistance = 7 , blockSize = 7)



# for clarification goto wiki  "pyramid(image processing)"
# maxLevel divides the resolution by 2 to compress the resolution for better clearity, the smaller the image better the dectection
# Criteria count is the max no of iteration ,i.e 10 (ACCURACY)
# EPS  epsilon is an upper bound on the error of a floating point numbers = 0.03 (SPEED)


lk_params = dict(winSize = (200,200),maxLevel = 2 ,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT , 10,0.03) )



cap = cv2.VideoCapture(0)


ret , prev_frame = cap.read()


prev_gray = cv2.cvtColor(prev_frame ,cv2.COLOR_BGR2GRAY)


#PTS TP TRACK


prevPts = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)


mask = np.zeros_like(prev_frame)


while True:
    ret, frame = cap.read()
    
    
    
    frame_gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    
    
    nextPts ,status ,err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)
    
    
    good_new = nextPts[status==1]
    good_prev = prevPts[status == 1]
    
    
    
    for i ,(new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new , y_new = new.ravel()        
        x_prev , y_prev = prev.ravel()
        
        
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
        
    img = cv2.add(frame,mask)
    cv2.imshow('tracking',img)
    
    
    k = cv2.waitKey(30) & 0xFF
    
    if k == 27:
        break
    
    
    prev_gray = frame_gray.copy()
    
    prevPts = good_new.reshape(-1,1,2)
    
    
    
    
cv2.destroyAllWindows()
cap.release()
    
    







# below is Farnebacl(Method)


# cap = cv2.VideoCapture(0)


# ret , frame1 = cap.read()


# prevsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# hsv_mask = np.zeros_like(frame1)
# hsv_mask[:,:,1] = 255


# while True:
    
    
#     ret,frame2 = cap.read()
    
#     nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
#     flow = cv2.calcOpticalFlowFarneback(prevsImg,nextImg,None,0.5,3,15,3,5,1.2,0)
    
#     # flow is in 'x' and 'y' coordinates , we need to cvt to magnitude and angle 
    
#     mag , ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True)
    
#     hsv_mask [:,:,0] = ang/2
    
    
#     hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
#     bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    
#     cv2.imshow('frame',bgr)
    
#     k =cv2.waitKey(10) & 0xFF
    
#     if k == 27:
#         break
        
#     prevsImg = nextImg
    
# cap.release()
# cv2.destroyAllWindows()
    

























