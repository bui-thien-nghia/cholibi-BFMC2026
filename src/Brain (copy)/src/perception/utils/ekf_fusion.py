import numpy as np
import time

class EKF_Fusion:
    """
    3-DOF Extended Kalman Filter for fusing IMU, Visual Odometry, and Landmarks.
    State Vector X = [x, y, theta, vx, vy, omega]^T
    """
    def __init__(self):
        # State Vector [x, y, theta, vx, vy, omega]
        self.x = np.zeros((6, 1))
        
        # Covariance Matrix P
        self.P = np.eye(6) * 0.1
        
        # Process Noise Q
        self.Q = np.eye(6) * 0.01
        
        # Measurement Noise R_vo (Visual Odometry)
        self.R_vo = np.eye(3) * 0.05
        
        # Measurement Noise R_landmark (Landmark Snap)
        self.R_landmark = np.eye(2) * 0.001 # High confidence
        
        self.last_time = time.time()

    def predict(self, accel_x, accel_y, gyro_z):
        """
        Prediction Step using IMU data.
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        theta = self.x[2, 0]
        vx = self.x[3, 0]
        vy = self.x[4, 0]
        
        # 1. State Prediction (Non-linear)
        # Simple constant velocity model + IMU acceleration
        # In car frame, vx is forward
        self.x[0, 0] += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        self.x[1, 0] += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        self.x[2, 0] += gyro_z * dt
        self.x[3, 0] += accel_x * dt
        self.x[4, 0] += accel_y * dt
        self.x[5, 0] = gyro_z
        
        # 2. Jacobian of the State Transition Matrix (F)
        F = np.eye(6)
        F[0, 2] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt
        F[0, 3] = np.cos(theta) * dt
        F[0, 4] = -np.sin(theta) * dt
        F[1, 2] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        F[1, 3] = np.sin(theta) * dt
        F[1, 4] = np.cos(theta) * dt
        F[2, 5] = dt
        F[3, 3] = 1 # simplified
        F[4, 4] = 1
        
        # 3. Covariance Prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_vo(self, dx, dy, dtheta):
        """
        Update Step using 2D Visual Odometry (delta measurement).
        Measurement Z = [dx, dy, dtheta]
        """
        # Measurement matrix H for VO
        # We observe deltas, so we can treat this as a velocity measurement * dt
        # Or more simply, update the state directly if dx, dy are global (but they are local)
        
        # Converting local VO movement to global update
        # For simplicity in this 3-DOF model, we treat VO as a direct state correction for velocity/heading
        
        z = np.array([[dx], [dy], [dtheta]])
        
        # Simplified measurement update for local deltas
        # In a real EKF, Z would be compared against h(X)
        # Here h(X) for local displacement would be [vx*dt, vy*dt, omega*dt]
        
        # Placeholder for proper VO measurement model
        pass

    def update_landmark(self, global_x, global_y):
        """
        Landmark Snap Update.
        Measurement Z = [global_x, global_y]
        """
        z = np.array([[global_x], [global_y]])
        
        # Measurement Matrix H (we observe x and y directly)
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        
        # Innovation (Residual)
        y = z - H @ self.x
        
        # Innovation Covariance
        S = H @ self.P @ H.T + self.R_landmark
        
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State Update
        self.x = self.x + K @ y
        
        # Covariance Update
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_state(self):
        return self.x.flatten()

    def set_state(self, x, y, theta):
        """
        Manually resets the EKF state to a specific pose.
        Used for map initialization.
        """
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = theta
        self.x[3:6, 0] = 0 # Reset velocities
        # Reset uncertainty for the new position
        self.P = np.eye(6) * 0.01 
