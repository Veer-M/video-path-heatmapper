import cv2
import numpy as np
from heatmapper import centroid, process

def test_detection_and_missing_marker():
    frame = np.zeros((80,100,3), np.uint8)
    assert centroid(frame) is None
    cv2.circle(frame, (30,40), 6, (0,0,255), -1)
    assert centroid(frame) == (30,40)

def test_video_pipeline(tmp_path):
    video = tmp_path/'sample.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100,80))
    assert writer.isOpened()
    for x in (20,30,40):
        frame = np.zeros((80,100,3), np.uint8)
        cv2.circle(frame,(x,40),6,(0,0,255),-1)
        writer.write(frame)
    writer.release()
    assert process(video,tmp_path/'out') == {'frames':3,'detections':3}
    assert cv2.imread(str(tmp_path/'out'/'heat_path.png')).sum() > 0
