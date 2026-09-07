from tracker import AttentionTracker
import os
t = AttentionTracker()
t.calibrate(duration=3)
with open('calib_done.flag', 'w') as flag: flag.write('1')
t.run()
