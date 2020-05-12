#!/usr/bin/env python2
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--op', type=str, help="Output location.", default='./landmarks')
parser.add_argument('vids', type=str, nargs='+', help="Input videos.")
args = parser.parse_args()

for vid in args.vids:
    init_time = time.clock()
    
    path = args.op + '/' + os.path.splitext(os.path.basename(vid))[0]
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    os.system("python vid2frame.py --op ./tmp " + vid)
    os.system("python landmark_extractor.py --lp " + path + ' ./tmp/*')

    os.system("rm -r ./tmp/")

    print("El video se ha procesado en: " + str(time.clock() - init_time))