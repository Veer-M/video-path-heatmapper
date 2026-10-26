import cv2
import numpy as np
writer = cv2.VideoWriter('gameplay.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (320,240))
if not writer.isOpened():
    raise RuntimeError('MJPG video encoder unavailable')
try:
    for i in range(180):
        frame = np.zeros((240,320,3), np.uint8)
        cv2.circle(frame,(25+i*3//2,120+int(60*np.sin(i/20))),8,(0,0,255),-1)
        writer.write(frame)
finally:
    writer.release()
print('Wrote gameplay.avi')
