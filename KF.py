"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""

import numpy as np

class KF_2D(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
            dt: sampling time (time for 1 cycle)
            u_x: acceleration in x-direction
            u_y: acceleration in y-direction
            std_acc: process noise magnitude
            x_std_meas: standard deviation of the measurement in x-direction
            y_std_meas: standard deviation of the measurement in y-direction
        """
        self.dt = dt # sampling time
        self.u = np.matrix([[u_x],[u_y]]) # control input variables
        self.x = np.matrix([[0], [0], [0], [0]]) # intial State
        
        # State Transition Matrix A (complete the definition)
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control Input Matrix B (defined for you)
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Measurement Mapping Matrix (complete the definition)
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance (defined for you)
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance (complete the definition)
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix (defined for you)
        self.P = np.eye(self.A.shape[1])


    def predict(self):
        ## complete this function
        # Update time state (self.x): x_k =Ax_(k-1) + Bu_(k-1) 
        # Calculate error covariance (self.P): P= A*P*A' + Q

        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0:2]

    def update(self, z):
        ## complete this function
        # Calculate S = H*P*H'+R
        # Calculate the Kalman Gain K = P * H'* inv(H*P*H'+R)
        # Update self.x
        # Update error covariance matrix self.P

        # Calculate S = H*P*H'+R
        # S = H*P*H'+R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Calculate the Kalman Gain K = P * H'* inv(H*P*H'+R)
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update self.x
        # x = x + K*(z - H*x)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        # Update error covariance matrix self.P
        # P = (I - K*H)*P
        self.P = np.dot((np.eye(self.H.shape[1]) - np.dot(K, self.H)), self.P)

        return self.x[0:2]

# 4D Kalman Filter: Track 4 D instead of 2D
# Track (x, y, w, h) instead of (x, y)

class KF_4D:
    def __init__(self, dt, u_x, u_y, u_w, u_h, std_acc, x_std_meas, y_std_meas, w_std_meas, h_std_meas):
        """
            dt: sampling time (time for 1 cycle)
            u_x: acceleration in x-direction
            u_y: acceleration in y-direction
            u_w: acceleration in w-direction
            u_h: acceleration in h-direction
            std_acc: process noise magnitude
            x_std_meas: standard deviation of the measurement in x-direction
            y_std_meas: standard deviation of the measurement in y-direction
            w_std_meas: standard deviation of the measurement in w-direction
            h_std_meas: standard deviation of the measurement in h-direction
        """
        self.dt = dt  # sampling time
        self.u = np.matrix([[u_x], [u_y], [u_w], [u_h]])  # control input variables
        self.x = np.matrix([[0], [0], [0], [0], [0], [0], [0], [0]])  # intial State

        # State Transition Matrix A
        self.A = np.matrix([[1, 0, 0, 0, self.dt, 0, 0, 0],
                            [0, 1, 0, 0, 0, self.dt, 0, 0],
                            [0, 0, 1, 0, 0, 0, self.dt, 0],
                            [0, 0, 0, 1, 0, 0, 0, self.dt],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]])

        # Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0, 0, 0],
                            [0,(self.dt**2)/2, 0, 0],
                            [0, 0, (self.dt**2)/2, 0],
                            [0, 0, 0, (self.dt**2)/2],
                            [self.dt, 0, 0, 0],
                            [0, self.dt, 0, 0],
                            [0, 0, self.dt, 0],
                            [0, 0, 0, self.dt]])

        # Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]])

        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0, (self.dt ** 2) / 2, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0, (self.dt ** 2) / 2, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, (self.dt ** 2), 0, self.dt, 0, 0, 0],
                            [0, (self.dt ** 3) / 2, 0, (self.dt ** 2), 0, self.dt, 0, 0],
                            [(self.dt ** 2) / 2, 0, self.dt, 0, 1, 0, 0, 0],
                            [0, (self.dt ** 2) / 2, 0, self.dt, 0, 1, 0, 0],
                            [(self.dt ** 3) / 2, 0, 0, 0, 0, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, 0, 0, 0, 0, self.dt ** 2]]) * std_acc ** 2

        self.R = np.matrix([[x_std_meas ** 2, 0, 0, 0],
                            [0, y_std_meas ** 2, 0, 0],
                            [0, 0, w_std_meas ** 2, 0],
                            [0, 0, 0, h_std_meas ** 2]])

        self.P = np.eye(self.A.shape[1])



    def predict(self):
        # Predict the state and covariance matrix

        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0:4]

    def update(self, z):
        # Update the state and covariance matrix based on measurement

        # S = H*P*H'+R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Calculate the Kalman Gain K = P * H'* inv(H*P*H'+R)
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update self.x
        # x = x + K*(z - H*x)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        # Update error covariance matrix self.P
        # P = (I - K*H)*P
        self.P = np.dot((np.eye(self.H.shape[1]) - np.dot(K, self.H)), self.P)

        return self.x[0:4]

