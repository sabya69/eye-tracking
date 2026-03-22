
from tracker import AttentionTracker

if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.calibrate(duration=3)
    tracker.run()
