"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""
import cv2
import numpy as np
from KF import KF_2D


def circleDetector(image):
    """ Simple OpenCV function for circle detection
        - detects edges, applies threshold to binary image space 
        - then find object countours 
        - then returns the center of the detected circle
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # get gray image
    img_edges = cv2.Canny(gray,  50, 190, 3) # detect edges
    #cv2.imshow('img_edges', img_edges)
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY) # convert to binary images
    #cv2.imshow('img_thresh', img_thresh)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get object contours

    # draw contour (see opencv tutorials)
    min_radius, max_radius = 3, 30
    centers=[]
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius) and (radius < max_radius): # Take only the valid circle(s)
            centers.append(np.array([[x], [y]]))
    cv2.imshow('contours', img_thresh)

    return centers




# OpenCV video capture object
VideoCap = cv2.VideoCapture('data/rBall.avi')

"""
# Create Kalman Filter object KF
    dt: sampling time (time for 1 cycle)
    u_x: acceleration in x-direction
    u_y: acceleration in y-direction
    std_acc: process noise magnitude
    x_std_meas: standard deviation of the measurement in x-direction
    y_std_meas: standard deviation of the measurement in y-direction
"""
filter = KF_2D(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

# Track circle (predict + update) using KF
while(True):
    ret, frame = VideoCap.read() # Read frame
    centers = circleDetector(frame) # Detect object
    # If centroids are detected then track them
    if (len(centers) > 0):
        cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2) # draw circle[0]
        cv2.putText(frame, "Measured Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,191,255), 2)

        # Predict
        (x, y) = filter.predict()
        cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2) # draw a rectangle
        cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)

        # Update
        (x1, y1) = filter.update(centers[0])
        cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2) # draw a rectangle
        cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)

    # Show output and wait for keypress
    cv2.imshow('image', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        break
    cv2.waitKey(25)

