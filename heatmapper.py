import argparse
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
from tracking import COLORS, Tracker, detect, track_statistics

def centroid(frame, min_area=10):
    detections = detect(frame,min_area=min_area)
    if not detections:
        return None
    largest = max(detections,key=lambda p:p['area'])
    return round(largest['x']),round(largest['y'])

def track_color(track_id):
    return tuple(int(v) for v in cv2.applyColorMap(np.uint8([[(track_id*43)%255]]),cv2.COLORMAP_TURBO)[0,0])

def save_image(path, image):
    if not cv2.imwrite(str(path),image):
        raise OSError(f'Cannot write {path}')

def process(video, output='results', min_area=10, color='red', max_area=None,
            max_distance=40, max_gap=12, roi=None, annotated=False, fps=None, max_frames=None):
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('FPS must be positive and finite')
    if max_frames is not None and max_frames < 1:
        raise ValueError('Frame limit must be positive')
    tracker = Tracker(max_distance,max_gap)
    capture = cv2.VideoCapture(str(video))
    writer = None
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frames,rows,detected_frames = 0,[],0
    path = None
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        fps = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no usable frame rate; supply --fps')
        while max_frames is None or frames < max_frames:
            ok,frame = capture.read()
            if not ok:
                break
            if path is None:
                height,width = frame.shape[:2]
                path = np.zeros((height,width),dtype=np.uint8)
                paths_by_id = np.zeros_like(frame)
                density = np.zeros((height,width),dtype=np.float32)
                first_frame = frame.copy()
                if annotated:
                    writer = cv2.VideoWriter(str(output/'annotated.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
                    if not writer.isOpened():
                        raise OSError('Cannot open MJPG annotated video writer')
            detections = detect(frame,color,min_area,max_area,roi)
            visible = tracker.update(detections,frames)
            detected_frames += bool(visible)
            for track_id,point in visible:
                x,y = round(point['x']),round(point['y'])
                cv2.circle(path,(x,y),2,255,-1)
                density[y,x] += 1/fps
                tint = track_color(track_id)
                points = tracker.tracks[track_id].points
                cv2.circle(paths_by_id,(x,y),2,tint,-1)
                if len(points) > 1 and points[-2]['frame'] == frames-1:
                    previous = (round(points[-2]['x']),round(points[-2]['y']))
                    cv2.line(paths_by_id,previous,(x,y),tint,2)
                rows.append((frames,frames/fps,track_id,point['x'],point['y'],point['area']))
                if writer is not None:
                    bx,by,bw,bh = point['bbox']
                    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),tint,1)
                    cv2.putText(frame,f'ID {track_id}',(bx,max(12,by-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,tint,1)
            if writer is not None:
                if roi is not None:
                    x,y,w,h = roi
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
                writer.write(cv2.addWeighted(frame,0.8,paths_by_id,0.6,0))
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if path is None:
        raise ValueError('Video contains no readable frames')
    blurred = cv2.GaussianBlur(density,(0,0),8)
    scaled = cv2.normalize(blurred,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
    heat[scaled == 0] = 0
    for filename,image in [('heat_path.png',path),('heat_density.png',heat),
                           ('tracks.png',paths_by_id),('overlay.png',cv2.addWeighted(first_frame,0.5,heat,0.7,0))]:
        save_image(output/filename,image)
    np.save(output/'occupancy_seconds.npy',density)
    with (output/'positions.csv').open('w',newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['frame','time_seconds','track_id','x','y','area_pixels'])
        csv_writer.writerows(rows)
    summary = {'frames':frames,'fps':fps,'duration_seconds':frames/fps,'detections':len(rows),
               'detected_frames':detected_frames,'frame_detection_fraction':detected_frames/frames,
               'tracks':track_statistics(tracker.tracks,fps),
               'configuration':{'color':color,'min_area':min_area,'max_area':max_area,
                                'max_distance':max_distance,'max_gap':max_gap,'roi':roi}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary

def main():
    parser = argparse.ArgumentParser(description='Track colored entities and measure movement in top-down video')
    parser.add_argument('video',nargs='?',default='gameplay.mp4')
    parser.add_argument('--output',default='results')
    parser.add_argument('--color',choices=COLORS,default='red')
    parser.add_argument('--min-area',type=int,default=10)
    parser.add_argument('--max-area',type=int)
    parser.add_argument('--max-distance',type=float,default=40)
    parser.add_argument('--max-gap',type=int,default=12)
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--annotated',action='store_true')
    parser.add_argument('--fps',type=float)
    parser.add_argument('--max-frames',type=int)
    args = parser.parse_args()
    try:
        result = process(**vars(args))
    except (ValueError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
