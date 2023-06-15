""" Autonomous Robots Kalman Filter 4D"""

import numpy as np
from KF import KF_4D
import cv2

def circleBBoxTracker(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray, 50, 190, 3)
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_radius, max_radius = 3, 30
    bboxes=[]
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius) and (radius < max_radius):
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append(np.array([[x], [y], [w], [h]]))

    return bboxes[0]

# OpenCV video capture object
VideoCap = cv2.VideoCapture('data/rBall.avi')

# Create Kalman Filter object KF
filter = KF_4D(dt=0.1, u_x=1, u_y=1, u_w=1, u_h=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1, w_std_meas=0.1, h_std_meas=0.1)

# Track circle (predict + update) using KF
while (True):
    ret, frame = VideoCap.read()  # Read frame
    bounding_box = circleBBoxTracker(frame)  # Detect object

    # If bounding box is detected then track it
    if bounding_box is not None:
        # Draw the bounding box
        cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])),
                      (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])),
                      (0, 255, 0),
                      2)  # draw a rectangle
        cv2.putText(frame, "Measured Position",
                    (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])), 0, 0.5,
                    (0, 255, 0), 2)

        # Predict
        x, y, w, h = filter.predict()
        predicted_box = (int(x - w / 2), int(y - h / 2), int(w), int(h))
        cv2.rectangle(frame, (int(x - w/2 ), int(y - h/2 )), (int(x + 2*w ), int(y + 2*h)), (255, 0, 0),
                      2)  # draw a rectangle
        cv2.putText(frame, "Predicted Position", (int(x + 2*w), int(y + 2*h)), 0, 0.5, (255, 0, 0), 2)


        # Update
        x1, y1, w1, h1 = filter.update(bounding_box)
        estimated_box = (int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1))
        cv2.rectangle(frame, (int(x1 - w1/2 ), int(y1 - h1/2 )), (int(x1 + 2*w1 ), int(y1 + 2*h1 )), (0, 0, 255),
                      2)  # draw a rectangle
        cv2.putText(frame, "Estimated Position", (int(x1 + 2*w1 ), int(y1 + 2*h1 )), 0, 0.5, (0, 0, 255), 2)


    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        break
    cv2.waitKey(25)
