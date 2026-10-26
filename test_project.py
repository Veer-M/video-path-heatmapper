import csv
import json
import cv2
import numpy as np
import pytest
from heatmapper import centroid, process
from tracking import Tracker, detect, track_statistics
from sample_video import generate

def point(x,y=20):
    return {'x':float(x),'y':float(y),'area':40,'bbox':(int(x)-2,int(y)-2,4,4)}

def test_detection_and_missing_marker():
    frame = np.zeros((80,100,3),np.uint8)
    assert centroid(frame) is None
    cv2.circle(frame,(30,40),6,(0,0,255),-1)
    assert centroid(frame) == (30,40)
    assert detect(frame,roi=(50,0,50,80)) == []

def test_hue_wrap_noise_and_area():
    hsv = np.zeros((80,100,3),np.uint8)
    cv2.circle(hsv,(20,20),6,(179,255,255),-1)
    cv2.circle(hsv,(60,40),1,(0,255,255),-1)
    frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    assert len(detect(frame)) == 1
    assert detect(frame,max_area=30) == []
    with pytest.raises(ValueError):
        detect(frame,roi=(90,0,50,20))

def test_ids_survive_order_changes_and_short_occlusion():
    tracker = Tracker(max_distance=10,max_gap=2)
    assert [t for t,_ in tracker.update([point(10),point(90)],0)] == [1,2]
    tracker.update([point(85),point(15)],1)
    tracker.update([],2)
    visible = tracker.update([point(75),point(25)],3)
    assert {t:p['x'] for t,p in visible} == {1:25,2:75}
    stats = track_statistics(tracker.tracks,10)
    assert stats[0]['distance_pixels'] == pytest.approx(15)
    assert stats[0]['mean_speed_pixels_per_second'] == pytest.approx(50)

def test_long_gap_creates_new_id():
    tracker = Tracker(max_gap=1)
    tracker.update([point(10)],0)
    assert tracker.update([point(10)],3)[0][0] == 2

def test_two_tracks_end_to_end(tmp_path):
    video = tmp_path/'sample.avi'
    generate(video)
    summary = process(video,tmp_path/'out',min_area=30,annotated=True)
    assert summary['frames'] == 180
    assert len(summary['tracks']) == 2
    assert summary['detections'] == 354
    assert summary['frame_detection_fraction'] == 1
    occupancy = np.load(tmp_path/'out'/'occupancy_seconds.npy')
    assert occupancy.sum() == pytest.approx(354/30,rel=1e-5)
    with (tmp_path/'out'/'positions.csv').open() as file:
        assert len(list(csv.DictReader(file))) == 354
    assert json.loads((tmp_path/'out'/'summary.json').read_text())['frames'] == 180
    annotated = cv2.VideoCapture(str(tmp_path/'out'/'annotated.avi'))
    assert annotated.isOpened()
    assert annotated.read()[0]
    annotated.release()

def test_no_detections_and_frame_limit(tmp_path):
    video = tmp_path/'sample.avi'
    generate(video,frames=10)
    summary = process(video,tmp_path/'out',color='green',max_frames=4)
    assert summary['frames'] == 4
    assert summary['tracks'] == []
    assert summary['detections'] == 0

def test_unreadable_input(tmp_path):
    with pytest.raises(ValueError):
        process(tmp_path/'missing.avi',tmp_path/'out')
