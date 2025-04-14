import json
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import ast
from scipy.spatial.distance import cdist
import cv2
import time
from datetime import datetime
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from collections import deque

@dataclass
class Appearance:
    embedding: np.ndarray
    quality_score: float
    timestamp: float
    camera_id: str

class KalmanTracker:
    def __init__(self):
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 0.1
        
        # Process noise
        self.kf.Q = np.eye(4) * 0.1
        self.kf.Q[2:, 2:] *= 0.1
        
        # Initial state covariance
        self.kf.P *= 1.0
        
    def initialize(self, position: np.ndarray):
        self.kf.x = np.array([position[0], position[1], 0, 0])
        
    def predict(self) -> np.ndarray:
        self.kf.predict()
        return self.kf.x[:2]
        
    def update(self, measurement: np.ndarray):
        self.kf.update(measurement)
        
    def get_state(self) -> np.ndarray:
        return self.kf.x[:2]
        
    def get_velocity(self) -> np.ndarray:
        return self.kf.x[2:]

class Detection:
    def __init__(self, sensor_id: str, frame_id: int, person_id: str, 
                 bbox: List[float], confidence: float, embedding: np.ndarray,
                 timestamp: float):
        self.sensor_id = sensor_id
        self.frame_id = frame_id
        self.person_id = person_id
        self.bbox = bbox
        self.confidence = confidence
        self.embedding = embedding
        self.timestamp = timestamp
        self.world_position = None

class Track:
    def __init__(self, detection: Detection, global_id: str):
        # Basic track information
        self.global_id = global_id
        self.detections = deque(maxlen=100)  # Recent detections
        self.appearances = []  # Long-term appearance history
        self.state = "active"  # active, dormant, inactive
        
        # Initialize Kalman filter
        self.kalman = KalmanTracker()
        self.kalman.initialize(detection.world_position)
        
        # Track statistics
        self.total_detections = 0  # Initialize to 0 since we'll increment in add_detection
        self.first_seen_time = detection.timestamp
        self.last_seen_time = detection.timestamp
        self.last_update_time = detection.timestamp
        self.camera_history = {}
        
        # Add the initial detection
        self.add_detection(detection)
        
    def add_detection(self, detection: Detection):
        # Add to recent detections queue
        self.detections.append(detection)
        
        # Update timing information
        self.last_update_time = detection.timestamp
        self.last_seen_time = detection.timestamp
        
        # Increment detection count
        self.total_detections += 1
        
        # Update camera history
        if detection.sensor_id in self.camera_history:
            self.camera_history[detection.sensor_id] += 1
        else:
            self.camera_history[detection.sensor_id] = 1
            
        # Update Kalman filter
        self.kalman.update(detection.world_position)
        
        # Add appearance if it's good quality
        quality_score = self._calculate_appearance_quality(detection)
        if quality_score > 0.7:  # Threshold for good quality
            appearance = Appearance(
                detection.embedding,
                quality_score,
                detection.timestamp,
                detection.sensor_id
            )
            self._update_appearances(appearance)
    
    def _calculate_appearance_quality(self, detection: Detection) -> float:
        # Combine multiple factors for quality score
        confidence_weight = 0.7
        size_weight = 0.3
        
        # Normalize bbox size
        bbox_size = (detection.bbox[2] * detection.bbox[3]) / (1920 * 1080)
        size_score = min(bbox_size * 10, 1.0)  # Scale up but cap at 1.0
        
        return (confidence_weight * detection.confidence + 
                size_weight * size_score)
    
    def _update_appearances(self, new_appearance: Appearance):
        # Keep max 10 best quality appearances
        self.appearances.append(new_appearance)
        self.appearances.sort(key=lambda x: x.quality_score, reverse=True)
        self.appearances = self.appearances[:10]
    
    def predict_position(self, timestamp: float) -> np.ndarray:
        """Predict position at given timestamp"""
        time_diff = timestamp - self.last_update_time
        # Multiple predictions for the time gap
        for _ in range(int(time_diff * 10)):  # 10 predictions per second
            self.kalman.predict()
        return self.kalman.get_state()
    
    def get_velocity(self) -> np.ndarray:
        return self.kalman.get_velocity()
    
    def update_state(self, current_time: float):
        """Update track state based on time"""
        time_since_update = current_time - self.last_update_time
        
        if time_since_update < 5.0:  # 5 seconds
            self.state = "active"
        elif time_since_update < 300.0:  # 5 minutes
            self.state = "dormant"
        else:
            self.state = "inactive"

class EnhancedMultiCameraTracker:
    def __init__(self, calibration: 'CameraCalibration', 
                 history_window: int = 300):  # 5 minutes
        self.calibration = calibration
        self.tracks: Dict[str, Track] = {}
        self.next_global_id = 0
        self.history_window = history_window
        
        # Tracking parameters
        self.max_embedding_dist = 0.8
        self.max_spatial_dist = {
            "active": 3.0,    # meters
            "dormant": 5.0,   # meters
            "inactive": 10.0  # meters
        }
        self.min_confidence_new_track = 0.3
        
    def calculate_similarity(self, track: Track, detection: Detection, 
                           current_time: float) -> float:
        """Calculate similarity between track and detection"""
        # Predict track position at detection time
        predicted_pos = track.predict_position(detection.timestamp)
        spatial_dist = np.linalg.norm(predicted_pos - detection.world_position)
        
        # Calculate best appearance similarity
        best_embedding_dist = float('inf')
        for appearance in track.appearances:
            embedding_similarity = np.dot(appearance.embedding, detection.embedding) / (
                np.linalg.norm(appearance.embedding) * np.linalg.norm(detection.embedding)
            )
            embedding_dist = 1 - embedding_similarity
            best_embedding_dist = min(best_embedding_dist, embedding_dist)
        
        # Get appropriate distance threshold based on track state
        max_spatial_dist = self.max_spatial_dist[track.state]
        
        # Calculate temporal penalty
        time_diff = detection.timestamp - track.last_update_time
        temporal_penalty = min(time_diff / 3600.0, 1.0)  # Max penalty after 1 hour
        
        # Combined score (lower is better)
        if (spatial_dist > max_spatial_dist or
            best_embedding_dist > self.max_embedding_dist):
            return float('inf')
            
        return (best_embedding_dist * 0.5 + 
                spatial_dist * 0.3 + 
                temporal_penalty * 0.2)
    
    def update(self, frame_detections: List[Detection], 
               current_time: float) -> List[Dict]:
        """Update tracks with new frame detections"""
        results = []
        
        # Update track states
        for track in self.tracks.values():
            track.update_state(current_time)
        
        # Sort detections by confidence
        frame_detections = sorted(frame_detections, key=lambda x: float(x['confidence']), reverse=True)
        unmatched_detections = frame_detections.copy()
        
        # Match with active tracks first
        matched_detections = set()
        active_tracks = [t for t in self.tracks.values() if t.state == "active"]
        
        for track in active_tracks:
            best_detection = None
            best_score = float('inf')
            
            for detection in unmatched_detections:
                score = self.calculate_similarity(track, detection, current_time)
                if score < best_score:
                    best_score = score
                    best_detection = detection
            
            if best_detection is not None:
                track.add_detection(best_detection)
                unmatched_detections.remove(best_detection)
                matched_detections.add(best_detection)
                
                results.append({
                    'global_person_id': track.global_id,
                    'person_id': best_detection.person_id,
                    'world_position': best_detection.world_position.tolist(),
                    'frame_id': best_detection.frame_id,
                    'state': track.state
                })
        
        # Try to match with dormant tracks
        dormant_tracks = [t for t in self.tracks.values() if t.state == "dormant"]
        for detection in unmatched_detections.copy():
            best_track = None
            best_score = float('inf')
            
            for track in dormant_tracks:
                score = self.calculate_similarity(track, detection, current_time)
                if score < best_score:
                    best_score = score
                    best_track = track
            
            if best_track is not None:
                best_track.add_detection(detection)
                best_track.state = "active"
                unmatched_detections.remove(detection)
                
                results.append({
                    'global_person_id': best_track.global_id,
                    'person_id': detection.person_id,
                    'world_position': detection.world_position.tolist(),
                    'frame_id': detection.frame_id,
                    'state': best_track.state
                })
        
        # Try to match with inactive tracks
        inactive_tracks = [t for t in self.tracks.values() if t.state == "inactive"]
        for detection in unmatched_detections.copy():
            if detection.confidence < self.min_confidence_new_track:
                continue
                
            best_track = None
            best_score = float('inf')
            
            for track in inactive_tracks:
                score = self.calculate_similarity(track, detection, current_time)
                if score < best_score:
                    best_score = score
                    best_track = track
            
            if best_track is not None:
                best_track.add_detection(detection)
                best_track.state = "active"
                unmatched_detections.remove(detection)
                
                results.append({
                    'global_person_id': best_track.global_id,
                    'person_id': detection.person_id,
                    'world_position': detection.world_position.tolist(),
                    'frame_id': detection.frame_id,
                    'state': best_track.state
                })
        
        # Create new tracks for remaining high-confidence detections
        for detection in unmatched_detections:
            if detection.confidence >= self.min_confidence_new_track:
                global_id = f"G{self.next_global_id}"
                self.next_global_id += 1
                
                track = Track(detection, global_id)
                self.tracks[global_id] = track
                
                results.append({
                    'global_person_id': global_id,
                    'person_id': detection.person_id,
                    'world_position': detection.world_position.tolist(),
                    'frame_id': detection.frame_id,
                    'state': track.state
                })
        
        return results



# In[2]:


#!pip install filterpy


# In[9]:


def write_final_summary(output_dir: str, tracker: EnhancedMultiCameraTracker, 
                       start_frame: int, end_frame: int, reason: str) -> None:
    """Write final tracking summary to file"""
    summary = {
        "total_frames_processed": end_frame - start_frame,
        "total_tracks": len(tracker.tracks),
        "track_statistics": {
            track_id: {
                "total_detections": track.total_detections,
                "first_seen_time": track.first_seen_time,
                "last_seen_time": track.last_seen_time,
                "camera_history": track.camera_history,
                "state": track.state
            }
            for track_id, track in tracker.tracks.items()
        },
        "end_frame": end_frame - 1,
        "end_time": datetime.now().isoformat(),
        "stop_reason": reason
    }
    
    with open(os.path.join(output_dir, "tracking_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

class CameraCalibration:
    def __init__(self, calibration_file: str):
        """Initialize camera calibration from JSON file"""
        with open(calibration_file, 'r') as f:
            self.calibration_data = json.load(f)
        
        # Pre-compute homography matrices for all cameras
        self.homography_matrices = {}
        self._compute_homography_matrices()
    
    def _compute_homography_matrices(self):
        """Pre-compute homography matrices for all cameras"""
        for sensor in self.calibration_data['sensors']:
            sensor_id = sensor['sensorId']
            
            image_points = np.array([[point['x'], point['y']] 
                                   for point in sensor['imageCoordinates']])
            
            world_points = np.array([[point['x'], point['y']] 
                                   for point in sensor['globalCoordinates']])
            
            H, _ = cv2.findHomography(image_points, world_points)
            self.homography_matrices[sensor_id] = H
    
    def image_to_world(self, camera_id: str, point: Tuple[float, float]) -> np.ndarray:
        """Convert image point to world coordinates using pre-computed homography"""
        H = self.homography_matrices[camera_id]
        point_homogeneous = np.array([point[0], point[1], 1])
        world_point = H @ point_homogeneous
        world_point = world_point / world_point[2]
        return world_point[:2]

def load_detections(file_path: str, calibration: CameraCalibration, 
                   current_time: float) -> Dict[str, List[Detection]]:
    """Load detections from a single frame file with world position calculation for each camera."""
    detections_by_camera = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = eval(line.strip())
                bbox = ast.literal_eval(data['bbox'])
                embedding = np.array(ast.literal_eval(data['embedding']))
                
                detection = Detection(
                    data['sensorId'],
                    int(data['frameId']),
                    data['personId'],
                    bbox,
                    float(data['confidence']),
                    embedding,
                    current_time
                )
                
                # Calculate world position using bottom center of bbox
                image_point = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3])
                detection.world_position = calibration.image_to_world(
                    detection.sensor_id, image_point
                )
                
                # Store detections grouped by camera
                if detection.sensor_id not in detections_by_camera:
                    detections_by_camera[detection.sensor_id] = []
                detections_by_camera[detection.sensor_id].append(detection)
        
        return detections_by_camera
    except FileNotFoundError:
        return {}


def process_frames(data_dir: str, calibration_file: str, output_base_dir: str,
                  start_frame: int = 0, sleep_time: float = 0.1, 
                  max_retries: int = 5) -> None:
    """Process frames with enhanced tracking capabilities"""
    # Initialize tracker
    calibration = CameraCalibration(calibration_file)
    tracker = EnhancedMultiCameraTracker(calibration)
    
    # Create output directory
    output_dir = setup_output_directory(output_base_dir)
    print(f"Saving results to: {output_dir}")
    
    # Save tracking parameters
    params = {
        "max_embedding_dist": tracker.max_embedding_dist,
        "max_spatial_dist": tracker.max_spatial_dist,
        "min_confidence_new_track": tracker.min_confidence_new_track,
        "start_frame": start_frame,
        "data_directory": data_dir,
        "calibration_file": calibration_file
    }
    
    with open(os.path.join(output_dir, "tracking_parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    
    current_frame = start_frame
    retry_count = 0
    start_time = time.time()
    
    print(f"Starting enhanced tracking from frame {start_frame}")
    
    try:
        while True:
            current_time = time.time() - start_time
            frame_file = f"{current_frame}.txt"
            frame_path = os.path.join(data_dir, frame_file)
            
            frame_detections = load_detections(frame_path, calibration, current_time)
            
            if frame_detections:
                print(f"Processing frame {current_frame}")
                results = tracker.update(frame_detections, current_time)
                
                # Add track states to results
                for result in results:
                    track = tracker.tracks[result['global_person_id']]
                    result['track_stats'] = {
                        'total_detections': track.total_detections,
                        'first_seen_time': track.first_seen_time,
                        'last_seen_time': track.last_seen_time,
                        'camera_history': track.camera_history,
                        'predicted_velocity': track.get_velocity().tolist()
                    }
                
                write_frame_results(output_dir, current_frame, results)
                
                current_frame += 1
                retry_count = 0
            else:
                retry_count += 1
                print(f"Frame {current_frame} not found. Retry {retry_count}/{max_retries}")
                
                if retry_count >= max_retries:
                    print(f"\nNo new frames found after {max_retries} attempts.")
                    write_final_summary(output_dir, tracker, start_frame, 
                                     current_frame, "max_retries_exceeded")
                    break
                
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nStopping tracking due to user interrupt...")
        write_final_summary(output_dir, tracker, start_frame, 
                          current_frame, "user_interrupt")
    
    except Exception as e:
        print(f"\nError during tracking: {e}")
        write_final_summary(output_dir, tracker, start_frame, 
                          current_frame, f"error: {str(e)}")
        raise


# In[10]:


def setup_output_directory(base_dir: str) -> str:
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"tracking_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def write_frame_results(output_dir: str, frame_id: int, results) -> None:
    """Write frame results to a JSON file"""
    output_file = os.path.join(output_dir, f"frame_{frame_id}.json")
       
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


# In[11]:


process_frames("C:/Users/Abhishek Talole/Desktop/Infinitraq/traker_kube_abhishek/track/preds/Building_K_Cam1/", "calibration_building_k_edit.json", "global_preds/", 4, 4 , 5)


# In[21]:




