import argparse
import cv2
import numpy as np

def generate(output='gameplay.avi', frames=180):
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
    if not writer.isOpened():
        raise OSError('MJPG video encoder unavailable')
    try:
        for i in range(frames):
            frame = np.full((240,320,3),20,np.uint8)
            for x in range(0,320,40):
                cv2.line(frame,(x,0),(x,239),(35,35,35),1)
            for y in range(0,240,40):
                cv2.line(frame,(0,y),(319,y),(35,35,35),1)
            cv2.circle(frame,(25+int(260*i/max(1,frames-1)),65+int(25*np.sin(i/20))),8,(0,0,255),-1)
            if not 65 <= i <= 70:
                cv2.circle(frame,(290-int(255*i/max(1,frames-1)),175+int(20*np.cos(i/18))),8,(0,0,255),-1)
            cv2.circle(frame,(155,120),1,(0,0,255),-1)
            writer.write(frame)
    finally:
        writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate two moving red markers with occlusion and noise')
    parser.add_argument('--output',default='gameplay.avi')
    args = parser.parse_args()
    generate(args.output)
    print(f'Wrote {args.output}')
