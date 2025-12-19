# FeelingDetector

An AI system designed to analyze facial expressions from images or live camera frames to infer human emotions.

## Overview

FeelingDetector analyzes facial features (eyes, eyebrows, mouth, facial tension) to detect primary emotions such as:
- **Happy, Sad, Angry, Anxious, Fearful**
- **Surprised, Disgusted, Confused, Calm, Neutral**

The system provides:
- **Detected Emotion(s)**: Primary emotions identified from facial features
- **Confidence Level**: Low / Medium / High
- **Facial Indicators Observed**: Bullet points of detected facial features
- **Explanation**: 1-2 sentences explaining the detection
- **Suggested Response**: A supportive or appropriate reply

## Features

- **Facial Landmark Detection**: Uses MediaPipe (recommended) or OpenCV for face detection
- **Multi-feature Analysis**: Analyzes mouth curvature, eyebrow position, eye openness
- **Real-time Camera Support**: Analyze emotions from webcam/live video feed
- **Image File Support**: Analyze emotions from static images (JPG, PNG, etc.)
- **Neutral and Respectful**: Does not identify or guess person's identity
- **Ambiguity Handling**: Clearly indicates when expressions are unclear

## Installation

1. Clone or download this repository
2. Ensure Python 3.7+ is installed
3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Verify installation (optional but recommended):

```bash
python test_mediapipe.py
```

### Dependencies

- **opencv-python** (>=4.8.0): Image processing and basic face detection
- **numpy** (>=1.24.0): Array operations
- **mediapipe** (>=0.10.0): **Required** - Advanced facial landmark detection with 468 landmarks

## Usage

### Command Line - Image File

Analyze an image file:

```bash
python feeling_detector.py path/to/image.jpg
```

### Command Line - Camera Mode

Run interactive camera mode:

```bash
python feeling_detector.py
```

Press:
- `q` to quit
- `s` to save and analyze current frame

### Programmatic Usage - Image File

```python
from feeling_detector import FeelingDetector

detector = FeelingDetector()

# Analyze image file
result = detector.detect_emotions(image_path="path/to/face_image.jpg")

# Print formatted output
print(detector.format_output(result))

# Or access individual components
print(f"Emotions: {result['emotions']}")
print(f"Confidence: {result['confidence']}")
print(f"Facial Indicators: {result['facial_indicators']}")
print(f"Explanation: {result['explanation']}")
print(f"Suggested Response: {result['suggested_response']}")
```

### Programmatic Usage - NumPy Array

```python
import cv2
from feeling_detector import FeelingDetector

detector = FeelingDetector()

# Load image as numpy array
image = cv2.imread("path/to/image.jpg")

# Analyze
result = detector.detect_emotions(image_array=image)
print(detector.format_output(result))
```

### Programmatic Usage - Camera Frame

```python
import cv2
from feeling_detector import FeelingDetector

detector = FeelingDetector()
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Analyze current frame
    result = detector.analyze_camera_frame(frame)
    
    # Display results
    print(f"Emotion: {result['emotions']}")
    print(f"Confidence: {result['confidence']}")
    
    # Your code to display frame, etc.
    
cap.release()
```

## Example Output

```
Detected Emotion(s): Happy
Confidence Level: High
Facial Indicators Observed:
- Upward mouth curvature (smiling)
- Neutral eyebrow position
- Normal eye openness
Explanation: The upward mouth curvature and relaxed facial muscles strongly indicate happy.
Suggested Response: You seem happy right now—hope whatever's going on continues to feel good.
```

## How It Works

1. **Face Detection**: Detects face in image using MediaPipe (preferred) or OpenCV Haar Cascade
2. **Landmark Extraction**: Extracts key facial landmarks (eyes, eyebrows, mouth, nose)
3. **Feature Analysis**:
   - **Mouth**: Analyzes curvature (upward = happy, downward = sad)
   - **Eyebrows**: Detects position (raised = surprised/anxious, lowered = angry)
   - **Eyes**: Measures openness (wide = surprised/fearful, narrow = angry/disgusted)
4. **Emotion Scoring**: Calculates emotion scores based on feature combinations
5. **Confidence Calculation**: Determines confidence based on feature clarity
6. **Explanation Generation**: Creates human-readable explanations
7. **Response Suggestion**: Generates empathetic responses

## Supported Emotions

- **Happy**: Upward mouth, relaxed features
- **Sad**: Downward mouth, lowered features
- **Angry**: Furrowed eyebrows, narrowed eyes, tense mouth
- **Anxious**: Raised eyebrows, tense features
- **Fearful**: Wide eyes, raised eyebrows
- **Surprised**: Wide eyes, raised eyebrows, open mouth
- **Disgusted**: Narrowed eyes, wrinkled nose
- **Confused**: Mixed or unclear features
- **Calm**: Neutral, relaxed features
- **Neutral**: Balanced, expressionless features

## Limitations

- Requires clear, front-facing face in image
- Works best with good lighting and clear visibility
- May struggle with:
  - Side profiles or angled faces
  - Very subtle expressions
  - Partially obscured faces
  - Low resolution images
- Accuracy depends on face detection quality
- Does not identify individuals (privacy-focused)

## Important Notes

⚠️ **Privacy & Ethics**:
- The AI does **not** identify or guess the person's identity
- Only analyzes visible facial expressions
- Does not make assumptions beyond what facial expressions reasonably suggest
- Respectful and non-judgmental analysis

⚠️ **Real-World Use**:
- The AI does not access the camera itself
- It only analyzes images or frames **provided to it**
- Best used with:
  - Webcam frame capture
  - Face landmark detection (MediaPipe, OpenCV)
  - Then send the image/frame to the AI

## Optional Enhancements

The following enhancements can be added upon request:

- ✅ Emotion intensity scoring (1-10 scale)
- ✅ Tuned for mental health support applications
- ✅ Combined analysis (face + text + voice)
- ✅ JSON-only output format
- ✅ Alignment with MediaPipe/OpenCV landmarks
- ✅ Real-time emotion tracking over time
- ✅ Multi-face detection and analysis

## Examples

See `example_facial_analysis.py` for comprehensive usage examples including:
- Image file analysis
- Camera/webcam analysis
- NumPy array analysis
- Batch image processing

Run examples:
```bash
python example_facial_analysis.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to:
- Use the software commercially
- Modify the software
- Distribute the software
- Sublicense the software
- Use it privately

The only requirement is that you include the original copyright notice and license text.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
