# motion.py
# -------------
# Interfaces and implementations for motion models (e.g., Kalman Filter)

from abc import ABC, abstractmethod
import numpy as np

class MotionModel(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray, cov: np.ndarray, dt: float):
        """
        Predict state and covariance forward by dt.
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        """
        Update state and covariance with a measurement.
        """
        pass

class GeneralKalmanMotionModel(MotionModel):
    """
    A generic Kalman Filter motion model. Parameterized by:
      - F_fn(dt): state transition function
      - Q_fn(dt): process noise covariance function
      - H: measurement matrix
      - R: measurement noise covariance
    """
    def __init__(self, F_fn, Q_fn, H: np.ndarray, R: np.ndarray):
        self.F_fn = F_fn  # Function to generate F given dt
        self.Q_fn = Q_fn  # Function to generate Q given dt
        self.H = H        # Matrix mapping state to measurement space
        self.R = R        # Measurement noise covariance

    def predict(self, state: np.ndarray, cov: np.ndarray, dt: float):
        # Build F and Q for this time interval
        F = self.F_fn(dt)
        Q = self.Q_fn(dt)
        # Predict next state mean and covariance
        state_pred = F @ state
        cov_pred = F @ cov @ F.T + Q
        return state_pred, cov_pred

    def update(self, state: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        # Compute innovation (measurement residual)
        y = measurement - self.H @ state
        # Innovation covariance
        S = self.H @ cov @ self.H.T + self.R
        # Kalman gain

        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            invS = np.linalg.pinv(S)

        K = cov @ self.H.T @ invS
        # Updated state estimate
        state_upd = state + K @ y
        # Updated covariance estimate
        I = np.eye(state.shape[0])
        cov_upd = (I - K @ self.H) @ cov
        return state_upd, cov_upd

# Factory function to create a Constant-Velocity Kalman

def make_CV_KalmanModel(process_noise: float = 1.0, measurement_noise: float = 1.0):
    """
    Returns a GeneralKalmanMotionModel configured for a 4D CV model ([x,y,vx,vy]).
    """
    # State transition includes dt into velocity terms
    def F_fn(dt: float):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    # Process noise for continuous white noise acceleration
    def Q_fn(dt: float):
        q = process_noise
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        return np.array([
            [dt4/4,    0, dt3/2,    0],
            [   0, dt4/4,    0, dt3/2],
            [dt3/2,    0,   dt2,    0],
            [   0, dt3/2,    0,   dt2]
        ], dtype=float) * q
    # Full observation of state
    H = np.eye(4, dtype=float)
    R = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float) * measurement_noise
    return GeneralKalmanMotionModel(F_fn, Q_fn, H, R)