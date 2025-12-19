"""
Example usage of FeelingDetector
Demonstrates various use cases and test cases
"""

from feeling_detector import FeelingDetector


def example_1():
    """Example 1: Simple happy emotion"""
    print("=" * 60)
    print("Example 1: Happy Emotion")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm so happy and excited about my new job!"
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_2():
    """Example 2: Anxious emotion"""
    print("=" * 60)
    print("Example 2: Anxious Emotion")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm really worried about the exam tomorrow. I feel so anxious and stressed."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_3():
    """Example 3: Multiple emotions"""
    print("=" * 60)
    print("Example 3: Multiple Emotions")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm frustrated because I can't figure this out, but I'm also hopeful that I'll get it soon."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_4():
    """Example 4: Negated emotion"""
    print("=" * 60)
    print("Example 4: Negated Emotion")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm not happy about this situation at all."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_5():
    """Example 5: Sad and lonely"""
    print("=" * 60)
    print("Example 5: Sad and Lonely")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I feel so sad and lonely. Nobody seems to understand what I'm going through."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_6():
    """Example 6: Angry and frustrated"""
    print("=" * 60)
    print("Example 6: Angry and Frustrated")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "This is so frustrating! I'm really angry that this keeps happening."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_7():
    """Example 7: Calm and hopeful"""
    print("=" * 60)
    print("Example 7: Calm and Hopeful")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm feeling calm and peaceful today. I'm hopeful that things will work out."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def example_8():
    """Example 8: Confused emotion"""
    print("=" * 60)
    print("Example 8: Confused Emotion")
    print("=" * 60)
    
    detector = FeelingDetector()
    text = "I'm really confused about what to do next. This doesn't make sense to me."
    result = detector.detect_emotions(text)
    print(f"Input: {text}\n")
    print(detector.format_output(result))
    print()


def run_all_examples():
    """Run all example cases"""
    examples = [
        example_1, example_2, example_3, example_4,
        example_5, example_6, example_7, example_8
    ]
    
    for example_func in examples:
        example_func()
        print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FeelingDetector - Example Usage")
    print("=" * 60 + "\n")
    
    run_all_examples()
    
    print("\n" + "=" * 60)
    print("Try your own text with: python feeling_detector.py")
    print("=" * 60)

