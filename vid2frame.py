import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vids', type=str, nargs='+', help="Input videos.")
parser.add_argument('outPath', type=str, help="Output location.")
args = parser.parse_args()

for vid in args.vids:

    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0

    try:
        os.mkdir(args.outPath)
    except OSError:
        print ("Creation of the directory %s failed" % args.outPath)
    else:
        print ("Successfully created the directory %s " % args.outPath)

    while success:
        cv2.imwrite(args.outPath + "/frame%d.jpg" % count, image)  
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1