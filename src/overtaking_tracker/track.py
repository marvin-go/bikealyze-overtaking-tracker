import numpy as np

class Track:
    """
    Represents a single tracked object in global UTM (x, y, vx, vy) with unique ID,
    state estimate, covariance of that estimate, and history of assigned detections.
    """
    def __init__(self, track_id: int, initial_det: dict, timestamp: float):
        # Unique identifier for this track
        self.track_id = track_id
        # Initialize state [x, y, vx, vy] from the first detection
        self.state = self._init_state(initial_det)
        # Initialize covariance matrix for uncertainty
        self.covariance = self._init_covariance()
        # Timestamp of the last update (in seconds)
        self.last_timestamp = timestamp
        # Counters used by the lifecycle manager
        self.hits = 1           # Number of frames this track was successfully matched
        self.misses = 0         # Consecutive frames without a match
        # Store original detection dicts to reconstruct the trajectory
        self.history = [initial_det]

    def _init_state(self, det: dict) -> np.ndarray:
        # Extract position and velocity from detection dict
        x, y = det['global_position'][:2]
        vx, vy = det['global_velocity'][:2]
        # Return as float array
        return np.array([x, y, vx, vy], dtype=float)

    def _init_covariance(self) -> np.ndarray:
        # Start with high uncertainty in both position and velocity
        return np.eye(4) * 1e3

    def predict(self, dt: float, motion_model) -> np.ndarray:
        """
        Predict the next state and covariance after time dt using the provided motion_model.
        Updates internal state and covariance, then returns the predicted state.
        """
        # Delegate Kalman predict step
        self.state, self.covariance = motion_model.predict(self.state, self.covariance, dt)
        # Advance timestamp
        self.last_timestamp += dt
        # Return only the mean for use in association
        return self.state

    def update(self, det: dict, motion_model) -> None:
        """
        Correct the state and covariance with a new detection.
        Resets miss count, increments hit count, and appends detection to history.
        """
        # Build measurement vector [x, y, vx, vy]
        meas = np.concatenate([det['global_position'][:2], det['global_velocity'][:2]]).astype(float)
        # Delegate Kalman update step
        self.state, self.covariance = motion_model.update(self.state, self.covariance, meas)
        # Lifecycle counters
        self.hits += 1
        self.misses = 0
        # Append detection for trajectory
        self.history.append(det)