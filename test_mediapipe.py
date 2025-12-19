"""
Test script to verify MediaPipe installation and functionality
"""

import sys

def test_mediapipe_import():
    """Test if MediaPipe can be imported."""
    try:
        import mediapipe as mp
        print("[OK] MediaPipe imported successfully")
        print(f"  Version: {mp.__version__ if hasattr(mp, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import MediaPipe: {e}")
        print("  Please install with: pip install mediapipe")
        return False

def test_mediapipe_face_mesh():
    """Test if MediaPipe Face Mesh can be initialized."""
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("[OK] MediaPipe Face Mesh initialized successfully")
        print("  - 468 facial landmarks available")
        print("  - Iris landmarks enabled (refine_landmarks=True)")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to initialize Face Mesh: {e}")
        return False

def test_feeling_detector():
    """Test if FeelingDetector can be initialized with MediaPipe."""
    try:
        from feeling_detector import FeelingDetector
        detector = FeelingDetector(use_mediapipe=True)
        
        if detector.is_mediapipe_available():
            print("[OK] FeelingDetector initialized with MediaPipe")
            print("  - Face Mesh: Active")
            print("  - Landmark detection: 468 points")
            return True
        else:
            print("[FAIL] FeelingDetector initialized but MediaPipe not available")
            return False
    except Exception as e:
        print(f"[FAIL] Failed to initialize FeelingDetector: {e}")
        return False

def test_opencv():
    """Test if OpenCV is available."""
    try:
        import cv2
        print(f"[OK] OpenCV imported successfully (version: {cv2.__version__})")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import OpenCV: {e}")
        return False

def test_numpy():
    """Test if NumPy is available."""
    try:
        import numpy as np
        print(f"[OK] NumPy imported successfully (version: {np.__version__})")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import NumPy: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MediaPipe Installation and Functionality Test")
    print("=" * 60)
    print()
    
    results = []
    
    print("Testing dependencies...")
    results.append(("NumPy", test_numpy()))
    print()
    results.append(("OpenCV", test_opencv()))
    print()
    results.append(("MediaPipe Import", test_mediapipe_import()))
    print()
    
    if results[-1][1]:  # If MediaPipe import succeeded
        results.append(("MediaPipe Face Mesh", test_mediapipe_face_mesh()))
        print()
        results.append(("FeelingDetector with MediaPipe", test_feeling_detector()))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    print()
    if all_passed:
        print("[SUCCESS] All tests passed! MediaPipe is ready to use.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please install missing dependencies.")
        print("\nInstallation command:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

