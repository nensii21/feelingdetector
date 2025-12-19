"""
Example usage of FeelingDetector for facial expression analysis
Demonstrates image analysis and camera frame analysis
"""

from feeling_detector import FeelingDetector
import cv2
import numpy as np


def example_image_analysis():
    """Example: Analyze emotion from an image file."""
    print("=" * 60)
    print("Example 1: Image File Analysis")
    print("=" * 60)
    
    detector = FeelingDetector()
    
    # Note: Replace with actual image path
    image_path = "test_face.jpg"  # Change this to your image path
    
    try:
        result = detector.detect_emotions(image_path=image_path)
        print(f"\nAnalyzing: {image_path}\n")
        print(detector.format_output(result))
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Please provide a valid image path with a face in it.")
    print()


def example_camera_analysis():
    """Example: Analyze emotions from webcam feed."""
    print("=" * 60)
    print("Example 2: Camera/Webcam Analysis")
    print("=" * 60)
    print("\nStarting camera...")
    print("Press 'q' to quit, 's' to analyze current frame\n")
    
    detector = FeelingDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_count = 0
    last_analysis = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze every 30 frames (approximately once per second at 30fps)
        if frame_count % 30 == 0:
            last_analysis = detector.analyze_camera_frame(frame)
        
        # Display current analysis on frame
        if last_analysis:
            emotion_text = ', '.join(last_analysis['emotions']) if last_analysis['emotions'] else 'Analyzing...'
            confidence_text = last_analysis['confidence']
            
            # Draw text on frame
            cv2.putText(frame, f"Emotion: {emotion_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence_text}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show top indicator
            if last_analysis['facial_indicators']:
                indicator = last_analysis['facial_indicators'][0]
                cv2.putText(frame, f"Indicator: {indicator[:40]}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('FeelingDetector - Press q to quit, s to save', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save and analyze current frame
            cv2.imwrite(f'captured_frame_{frame_count}.jpg', frame)
            result = detector.analyze_camera_frame(frame)
            print("\n" + detector.format_output(result) + "\n")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed.")


def example_numpy_array():
    """Example: Analyze emotion from numpy array (e.g., from PIL or other sources)."""
    print("=" * 60)
    print("Example 3: NumPy Array Analysis")
    print("=" * 60)
    
    detector = FeelingDetector()
    
    # Create a dummy image array (in real use, this would come from PIL, etc.)
    # This is just for demonstration
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image.fill(128)  # Gray image
    
    print("\nNote: This is a dummy image (gray). In real usage, provide an actual face image.")
    result = detector.detect_emotions(image_array=dummy_image)
    print(detector.format_output(result))
    print()


def example_batch_analysis():
    """Example: Analyze multiple images."""
    print("=" * 60)
    print("Example 4: Batch Image Analysis")
    print("=" * 60)
    
    detector = FeelingDetector()
    
    # List of image paths to analyze
    image_paths = [
        "face1.jpg",
        "face2.jpg",
        "face3.jpg"
    ]
    
    print("\nAnalyzing multiple images...\n")
    
    for image_path in image_paths:
        try:
            result = detector.detect_emotions(image_path=image_path)
            print(f"Image: {image_path}")
            print(f"  Emotions: {', '.join(result['emotions']) if result['emotions'] else 'None'}")
            print(f"  Confidence: {result['confidence']}")
            print()
        except Exception as e:
            print(f"  Error analyzing {image_path}: {e}\n")
    
    print("Batch analysis complete.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FeelingDetector - Facial Expression Analysis Examples")
    print("=" * 60 + "\n")
    
    print("Choose an example to run:")
    print("1. Image file analysis")
    print("2. Camera/webcam analysis (interactive)")
    print("3. NumPy array analysis")
    print("4. Batch image analysis")
    print("5. Run all examples")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        example_image_analysis()
    elif choice == "2":
        example_camera_analysis()
    elif choice == "3":
        example_numpy_array()
    elif choice == "4":
        example_batch_analysis()
    elif choice == "5":
        example_image_analysis()
        example_numpy_array()
        example_batch_analysis()
        print("\nNote: Camera example requires manual interaction, run separately.")
    else:
        print("Invalid choice. Running image analysis example...")
        example_image_analysis()

