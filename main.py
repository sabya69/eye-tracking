# ============================================================
#  FILE: main.py
#  This is the ONLY file you run.
#
#  In VS Code terminal:
#      pip install -r requirements.txt
#      python main.py
# ============================================================

from tracker import AttentionTracker

if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.calibrate(duration=3)
    tracker.run()
