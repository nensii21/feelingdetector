"""
FeelingDetector - An AI designed to analyze facial expressions from images or live camera frames to infer human emotions.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json


class FeelingDetector:
    """
    An AI system that analyzes facial expressions from images to detect and explain human emotions.
    """
    
    # MediaPipe face mesh landmark indices for key facial features
    # MediaPipe Face Mesh provides 468 3D facial landmarks
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LANDMARK_INDICES = {
        # Left eye (outer to inner)
        'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        # Right eye (outer to inner)
        'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        # Left eyebrow (outer to inner)
        'left_eyebrow': [70, 63, 105, 66, 107],
        # Right eyebrow (outer to inner)
        'right_eyebrow': [336, 296, 334, 293, 300],
        # Mouth outline
        'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        # Mouth corners: left (61) and right (291)
        'mouth_corners': [61, 291],
        # Upper lip center
        'upper_lip': [13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
        # Lower lip center
        'lower_lip': [14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
        # Nose tip
        'nose_tip': [4],
        # Nose bridge
        'nose_bridge': [6, 168, 8, 9, 10],
        # Jawline
        'jaw': [10, 151, 9, 175, 199]
    }
    
    def __init__(self, use_mediapipe: bool = True):
        """
        Initialize the FeelingDetector.
        
        Args:
            use_mediapipe: If True, use MediaPipe for face detection (requires mediapipe package)
                          If False, use OpenCV's Haar Cascade (built-in, less accurate)
        """
        self.use_mediapipe = use_mediapipe
        self.face_detector = None
        self.face_mesh = None
        
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp = mp
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                
                # Initialize face detection (for bounding boxes)
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.5
                )
                
                # Initialize face mesh (for 468 landmarks)
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,  # Enables iris landmarks for better eye detection
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                print("MediaPipe initialized successfully with Face Mesh (468 landmarks)")
            except ImportError:
                print("ERROR: MediaPipe is required but not installed.")
                print("Please install it with: pip install mediapipe")
                print("Falling back to OpenCV Haar Cascade (less accurate).")
                self.use_mediapipe = False
                self._init_opencv()
            except Exception as e:
                print(f"Warning: Error initializing MediaPipe: {e}")
                print("Falling back to OpenCV Haar Cascade.")
                self.use_mediapipe = False
                self._init_opencv()
        else:
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize OpenCV face detector as fallback."""
        try:
            # Try to load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise FileNotFoundError("Haar Cascade file not found")
        except Exception as e:
            print(f"Warning: Could not initialize OpenCV face detector: {e}")
            self.face_cascade = None
    
    def detect_emotions(self, image_path: Optional[str] = None, image_array: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze facial expression from an image and detect emotions.
        
        Args:
            image_path: Path to image file (jpg, png, etc.)
            image_array: numpy array of image (BGR format, as from cv2.imread)
            
        Returns:
            Dictionary containing detected emotions, confidence, facial indicators, explanation, and suggested response
        """
        # Load image
        if image_array is not None:
            image = image_array.copy()
        elif image_path:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'emotions': [],
                    'confidence': 'Low',
                    'facial_indicators': ['Could not load image'],
                    'explanation': 'Failed to load the image file. Please check the path and file format.',
                    'suggested_response': 'Please provide a valid image file.'
                }
        else:
            return {
                'emotions': [],
                'confidence': 'Low',
                'facial_indicators': ['No image provided'],
                'explanation': 'No image was provided for analysis.',
                'suggested_response': 'Please provide an image or image path to analyze.'
            }
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face and landmarks
        facial_features = self._extract_facial_features(image_rgb, image)
        
        if not facial_features:
            return {
                'emotions': [],
                'confidence': 'Low',
                'facial_indicators': ['No face detected'],
                'explanation': 'No face was detected in the image. Please ensure the image contains a clear, front-facing face.',
                'suggested_response': 'Could you provide an image with a visible face?'
            }
        
        # Analyze facial features to determine emotions
        emotions, confidence, indicators = self._analyze_facial_features(facial_features)
        
        # Generate explanation
        explanation = self._generate_explanation(emotions, indicators)
        
        # Generate suggested response
        suggested_response = self._generate_suggested_response(emotions)
        
        return {
            'emotions': emotions,
            'confidence': confidence,
            'facial_indicators': indicators,
            'explanation': explanation,
            'suggested_response': suggested_response
        }
    
    def _extract_facial_features(self, image_rgb: np.ndarray, image_bgr: np.ndarray) -> Optional[Dict]:
        """Extract facial features and landmarks from the image."""
        facial_features = {}
        
        if self.use_mediapipe and self.face_mesh:
            try:
                results = self.face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w = image_rgb.shape[:2]
                    
                    # Extract key points with MediaPipe landmarks
                    facial_features = {
                        'left_eye': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['left_eye'], w, h),
                        'right_eye': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['right_eye'], w, h),
                        'left_eyebrow': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['left_eyebrow'], w, h),
                        'right_eyebrow': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['right_eyebrow'], w, h),
                        'mouth': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['mouth'], w, h),
                        'mouth_corners': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['mouth_corners'], w, h),
                        'upper_lip': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES.get('upper_lip', []), w, h),
                        'lower_lip': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES.get('lower_lip', []), w, h),
                        'nose_tip': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES['nose_tip'], w, h),
                        'nose_bridge': self._get_landmark_points(face_landmarks, self.LANDMARK_INDICES.get('nose_bridge', []), w, h),
                    }
                    return facial_features
            except Exception as e:
                print(f"Error in MediaPipe processing: {e}")
        
        # Fallback: Use basic face detection and estimate features
        if self.face_cascade:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Estimate facial regions (rough approximation)
                facial_features = {
                    'face_bbox': (x, y, w, h),
                    'estimated': True
                }
                return facial_features
        
        return None
    
    def _get_landmark_points(self, landmarks, indices: List[int], width: int, height: int) -> List[Tuple[float, float]]:
        """
        Extract landmark points from MediaPipe results.
        
        Args:
            landmarks: MediaPipe face landmarks object
            indices: List of landmark indices to extract
            width: Image width
            height: Image height
            
        Returns:
            List of (x, y) coordinates in pixel space
        """
        points = []
        if not indices:
            return points
            
        for idx in indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                # MediaPipe returns normalized coordinates (0-1), convert to pixel coordinates
                x = landmark.x * width
                y = landmark.y * height
                points.append((x, y))
        return points
    
    def is_mediapipe_available(self) -> bool:
        """Check if MediaPipe is available and initialized."""
        return self.use_mediapipe and self.face_mesh is not None
    
    def _analyze_facial_features(self, features: Dict) -> Tuple[List[str], str, List[str]]:
        """
        Analyze facial features to determine emotions.
        
        Returns:
            Tuple of (emotions_list, confidence_level, facial_indicators_list)
        """
        indicators = []
        emotion_scores = {}
        
        # If using estimated features (fallback), provide basic analysis
        if features.get('estimated', False):
            return (['Neutral'], 'Low', ['Face detected but detailed analysis unavailable'])
        
        # Analyze mouth (most important for emotion)
        if 'mouth_corners' in features and len(features['mouth_corners']) >= 2:
            left_corner = features['mouth_corners'][0]
            right_corner = features['mouth_corners'][1] if len(features['mouth_corners']) > 1 else left_corner
            
            # Calculate mouth curvature using multiple methods for better accuracy
            mouth_curve = 0
            has_smile_data = False
            
            # Method 1: Compare corners to mouth center
            if 'mouth' in features and len(features['mouth']) > 0:
                mouth_points = features['mouth']
                mouth_center_y = sum(p[1] for p in mouth_points) / len(mouth_points)
                corner_avg_y = (left_corner[1] + right_corner[1]) / 2
                mouth_curve = corner_avg_y - mouth_center_y
                has_smile_data = True
            
            # Method 2: Compare upper and lower lip (more accurate)
            if 'upper_lip' in features and 'lower_lip' in features:
                if len(features['upper_lip']) > 0 and len(features['lower_lip']) > 0:
                    upper_lip_y = sum(p[1] for p in features['upper_lip']) / len(features['upper_lip'])
                    lower_lip_y = sum(p[1] for p in features['lower_lip']) / len(features['lower_lip'])
                    lip_separation = lower_lip_y - upper_lip_y
                    
                    # If corners are significantly above the lip center, it's a smile
                    corner_avg_y = (left_corner[1] + right_corner[1]) / 2
                    lip_center_y = (upper_lip_y + lower_lip_y) / 2
                    mouth_curve = corner_avg_y - lip_center_y
                    has_smile_data = True
            
            if has_smile_data:
                if mouth_curve < -5:  # Corners above center = smile
                    emotion_scores['Happy'] = 8
                    indicators.append("Upward mouth curvature (smiling)")
                elif mouth_curve > 5:  # Corners below center = frown
                    emotion_scores['Sad'] = 7
                    indicators.append("Downward mouth curvature (frowning)")
                else:
                    indicators.append("Neutral mouth position")
        
        # Analyze eyebrows
        if 'left_eyebrow' in features and 'right_eyebrow' in features:
            left_eyebrow = features['left_eyebrow']
            right_eyebrow = features['right_eyebrow']
            
            if left_eyebrow and right_eyebrow:
                left_avg_y = sum(p[1] for p in left_eyebrow) / len(left_eyebrow)
                right_avg_y = sum(p[1] for p in right_eyebrow) / len(right_eyebrow)
                
                # Compare with eye position
                if 'left_eye' in features and features['left_eye']:
                    eye_avg_y = sum(p[1] for p in features['left_eye']) / len(features['left_eye'])
                    eyebrow_eye_diff = left_avg_y - eye_avg_y
                    
                    if eyebrow_eye_diff < -10:  # Eyebrows raised
                        emotion_scores['Surprised'] = emotion_scores.get('Surprised', 0) + 5
                        emotion_scores['Anxious'] = emotion_scores.get('Anxious', 0) + 3
                        indicators.append("Raised eyebrows")
                    elif eyebrow_eye_diff > 15:  # Eyebrows lowered
                        emotion_scores['Angry'] = emotion_scores.get('Angry', 0) + 6
                        emotion_scores['Confused'] = emotion_scores.get('Confused', 0) + 2
                        indicators.append("Lowered/furrowed eyebrows")
                    else:
                        indicators.append("Neutral eyebrow position")
        
        # Analyze eyes
        if 'left_eye' in features and 'right_eye' in features:
            left_eye = features['left_eye']
            right_eye = features['right_eye']
            
            if left_eye and right_eye:
                # Check eye openness (simplified)
                left_height = max(p[1] for p in left_eye) - min(p[1] for p in left_eye)
                right_height = max(p[1] for p in right_eye) - min(p[1] for p in right_eye)
                avg_eye_height = (left_height + right_height) / 2
                
                if avg_eye_height > 15:  # Wide open eyes
                    emotion_scores['Surprised'] = emotion_scores.get('Surprised', 0) + 4
                    emotion_scores['Fearful'] = emotion_scores.get('Fearful', 0) + 3
                    indicators.append("Wide open eyes")
                elif avg_eye_height < 8:  # Narrow/squinted eyes
                    emotion_scores['Angry'] = emotion_scores.get('Angry', 0) + 3
                    emotion_scores['Disgusted'] = emotion_scores.get('Disgusted', 0) + 2
                    indicators.append("Narrowed/squinted eyes")
                else:
                    indicators.append("Normal eye openness")
        
        # Determine primary emotions
        if not emotion_scores:
            return (['Neutral'], 'Low', indicators if indicators else ['Limited facial feature data'])
        
        # Sort emotions by score
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top emotions
        detected_emotions = []
        top_score = sorted_emotions[0][1]
        for emotion, score in sorted_emotions:
            if score >= top_score * 0.6:  # Include emotions with at least 60% of top score
                detected_emotions.append(emotion)
            if len(detected_emotions) >= 2:
                break
        
        # Calculate confidence
        if top_score >= 7:
            confidence = 'High'
        elif top_score >= 4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return (detected_emotions, confidence, indicators)
    
    def _generate_explanation(self, emotions: List[str], indicators: List[str]) -> str:
        """Generate explanation for detected emotions."""
        if not emotions or emotions == ['Neutral']:
            return "The facial expression appears neutral or the emotional indicators are unclear."
        
        emotion_str = " and ".join(emotions) if len(emotions) > 1 else emotions[0]
        
        if 'Happy' in emotions:
            return f"The upward mouth curvature and relaxed facial muscles strongly indicate {emotion_str.lower()}."
        elif 'Sad' in emotions:
            return f"The downward mouth curvature and facial tension suggest {emotion_str.lower()}."
        elif 'Angry' in emotions:
            return f"The furrowed eyebrows and tense facial features indicate {emotion_str.lower()}."
        elif 'Surprised' in emotions:
            return f"The raised eyebrows and wide eyes suggest {emotion_str.lower()}."
        elif 'Anxious' in emotions or 'Fearful' in emotions:
            return f"The raised eyebrows and tense facial expression indicate {emotion_str.lower()}."
        else:
            return f"The observed facial features suggest {emotion_str.lower()}."
    
    def _generate_suggested_response(self, emotions: List[str]) -> str:
        """Generate a supportive suggested response."""
        if not emotions or emotions == ['Neutral']:
            return "Your expression appears neutral. How are you feeling?"
        
        emotion_lower = emotions[0].lower()
        
        responses = {
            'happy': "You seem happy right now—hope whatever's going on continues to feel good.",
            'sad': "You appear to be feeling down. I'm here if you'd like to talk about what's on your mind.",
            'angry': "You seem upset. Taking a moment to breathe might help.",
            'anxious': "You look a bit anxious. Remember to take things one step at a time.",
            'fearful': "You appear frightened or concerned. You're safe here.",
            'surprised': "You look surprised! Is something unexpected happening?",
            'disgusted': "You seem displeased or disgusted. What's bothering you?",
            'confused': "You look confused. Would it help to talk through what's unclear?",
            'calm': "You appear calm and relaxed. That's nice to see.",
            'neutral': "Your expression seems neutral. How are you feeling?"
        }
        
        return responses.get(emotion_lower, "I notice your expression. How can I support you right now?")
    
    def format_output(self, result: Dict) -> str:
        """
        Format the detection result in the specified output format.
        
        Args:
            result: Dictionary from detect_emotions()
            
        Returns:
            Formatted string output
        """
        emotions_str = ", ".join(result['emotions']) if result['emotions'] else "None detected"
        
        indicators_str = "\n".join(f"- {ind}" for ind in result['facial_indicators']) if result['facial_indicators'] else "- No indicators observed"
        
        output = f"""Detected Emotion(s): {emotions_str}
Confidence Level: {result['confidence']}
Facial Indicators Observed:
{indicators_str}
Explanation: {result['explanation']}
Suggested Response: {result['suggested_response']}"""
        
        return output
    
    def analyze_camera_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame from a camera/webcam.
        
        Args:
            frame: numpy array of image frame (BGR format from cv2.VideoCapture)
            
        Returns:
            Dictionary containing detected emotions and analysis
        """
        return self.detect_emotions(image_array=frame)


def main():
    """Main function for command-line usage."""
    import sys
    
    detector = FeelingDetector()
    
    print("=" * 60)
    print("FeelingDetector - Facial Expression Analysis System")
    print("=" * 60)
    print("\nUsage:")
    print("  python feeling_detector.py <image_path>")
    print("  or run without arguments for camera mode\n")
    
    if len(sys.argv) > 1:
        # Analyze provided image
        image_path = sys.argv[1]
        print(f"Analyzing image: {image_path}\n")
        result = detector.detect_emotions(image_path=image_path)
        print(detector.format_output(result))
    else:
        # Camera mode
        print("Starting camera mode...")
        print("Press 'q' to quit, 's' to save current frame\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 10th frame for performance
            if frame_count % 10 == 0:
                result = detector.analyze_camera_frame(frame)
                
                # Display results on frame
                cv2.putText(frame, f"Emotion: {', '.join(result['emotions']) if result['emotions'] else 'None'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {result['confidence']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('FeelingDetector - Press q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'captured_frame_{frame_count}.jpg', frame)
                print(f"Frame saved as captured_frame_{frame_count}.jpg")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nThank you for using FeelingDetector. Take care!")


if __name__ == "__main__":
    main()
