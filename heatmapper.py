import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

def centroid(frame, min_area=10):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
    mask |= cv2.inRange(hsv, np.array([170,120,70]), np.array([179,255,255]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    target = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[target, cv2.CC_STAT_AREA] < min_area:
        return None
    moments = cv2.moments((labels == target).astype(np.uint8))
    return (int(round(moments['m10']/moments['m00'])), int(round(moments['m01']/moments['m00'])))

def process(video, output='results', min_area=10):
    if min_area <= 0:
        raise ValueError('Minimum area must be positive')
    capture = cv2.VideoCapture(str(video))
    path = None
    rows = []
    index = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if path is None:
                path = np.zeros(frame.shape[:2], dtype=np.uint8)
                density = np.zeros(frame.shape[:2], dtype=np.float32)
            point = centroid(frame, min_area)
            if point is not None:
                cv2.circle(path, point, 2, 255, -1)
                density[point[1], point[0]] += 1
                rows.append((index, *point))
            index += 1
    finally:
        capture.release()
    if path is None:
        raise ValueError('Video contains no readable frames')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    blurred = cv2.GaussianBlur(density, (0,0), 8)
    scaled = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    heat[scaled == 0] = 0
    for filename, image in [('heat_path.png', path), ('heat_density.png', heat)]:
        if not cv2.imwrite(str(output / filename), image):
            raise OSError(f'Cannot write {filename}')
    with (output / 'positions.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame','x','y'])
        writer.writerows(rows)
    return {'frames': index, 'detections': len(rows)}

def main():
    parser = argparse.ArgumentParser(description='Track one red marker in a top-down video')
    parser.add_argument('video', nargs='?', default='gameplay.mp4')
    parser.add_argument('--output', default='results')
    parser.add_argument('--min-area', type=int, default=10)
    args = parser.parse_args()
    print(process(args.video, args.output, args.min_area))

if __name__ == '__main__':
    main()
