from dataclasses import dataclass, field
import math
import cv2
import numpy as np

COLORS = {
    'red': [((0,120,70),(10,255,255)), ((170,120,70),(179,255,255))],
    'green': [((35,80,50),(85,255,255))],
    'blue': [((95,100,50),(135,255,255))],
}

def detect(frame, color='red', min_area=10, max_area=None, roi=None):
    if color not in COLORS or min_area <= 0 or max_area is not None and max_area < min_area:
        raise ValueError('Choose a supported color and valid area limits')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2],dtype=np.uint8)
    for lower,upper in COLORS[color]:
        mask |= cv2.inRange(hsv,np.array(lower),np.array(upper))
    if roi is not None:
        x,y,w,h = roi
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > mask.shape[1] or y+h > mask.shape[0]:
            raise ValueError('ROI must be a positive rectangle inside the video')
        selected = np.zeros_like(mask)
        selected[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
        mask = selected
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,_,stats,centers = cv2.connectedComponentsWithStats(mask)
    detections = []
    for i in range(1,count):
        x,y,w,h,area = map(int,stats[i])
        if area < min_area or max_area is not None and area > max_area:
            continue
        detections.append({'x':float(centers[i][0]),'y':float(centers[i][1]),
                           'area':area,'bbox':(x,y,w,h)})
    return sorted(detections,key=lambda p:(p['x'],p['y']))

@dataclass
class Track:
    id: int
    points: list = field(default_factory=list)

    def predict(self, frame):
        last = self.points[-1]
        if len(self.points) < 2:
            return last['x'],last['y']
        before = self.points[-2]
        factor = (frame-last['frame'])/(last['frame']-before['frame'])
        return last['x']+(last['x']-before['x'])*factor,last['y']+(last['y']-before['y'])*factor

class Tracker:
    def __init__(self, max_distance=40, max_gap=12):
        if not math.isfinite(max_distance) or max_distance <= 0 or max_gap < 0:
            raise ValueError('Distance must be positive and max gap nonnegative')
        self.max_distance,self.max_gap = max_distance,max_gap
        self.tracks = {}
        self.next_id,self.last_frame = 1,-1

    def update(self, detections, frame):
        if frame <= self.last_frame:
            raise ValueError('Frame numbers must strictly increase')
        self.last_frame = frame
        active = [t for t in self.tracks.values() if frame-t.points[-1]['frame'] <= self.max_gap+1]
        candidates = []
        for track in active:
            px,py = track.predict(frame)
            elapsed = frame-track.points[-1]['frame']
            for index,p in enumerate(detections):
                distance = math.hypot(px-p['x'],py-p['y'])
                if distance <= self.max_distance*elapsed:
                    candidates.append((distance,track.id,index))
        matched_tracks,matched_detections = set(),set()
        assignments = []
        for _,track_id,index in sorted(candidates):
            if track_id in matched_tracks or index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(index)
            assignments.append((track_id,index))
        for index in range(len(detections)):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id)
                assignments.append((track_id,index))
        visible = []
        for track_id,index in assignments:
            point = {'frame':frame,**detections[index]}
            self.tracks[track_id].points.append(point)
            visible.append((track_id,point))
        return sorted(visible)

def track_statistics(tracks, fps):
    result = []
    for track in tracks.values():
        points = track.points
        distance = sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(points,points[1:]))
        span = (points[-1]['frame']-points[0]['frame'])/fps
        result.append({'id':track.id,'detections':len(points),'first_frame':points[0]['frame'],
                       'last_frame':points[-1]['frame'],'span_seconds':span,
                       'observed_seconds':len(points)/fps,'distance_pixels':distance,
                       'mean_speed_pixels_per_second':distance/span if span else 0.0})
    return result
