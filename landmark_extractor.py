#!/usr/bin/env python2

import json

import argparse
import cv2
import itertools
import os
import time
import numpy as np
np.set_printoptions(precision=2)
import openface

init_time = time.time()
fileDir = '/root/openface'
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--lp', type=str, default='./landmarks' , help="Path where landmarks would be stored.")
args = parser.parse_args()

imgDim = 96
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), imgDim)



def getLandmarks(imgPath):

    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    
    if bb is None:
        print("Unable to find a face: {}".format(imgPath))
    else:

        # if not os.path.isdir(args.lp):
        #     try:
        #         os.mkdir(args.lp)
        #     except OSError:
        #         print ("Creation of the directory %s failed" % args.lp)
        #     else:
        #         print ("Successfully created the directory %s " % args.lp)

        alignedFace = align.align(imgDim, rgbImg, bb,
                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            print("Unable to align image: {}".format(imgPath))
            return ''


        bb = align.getLargestFaceBoundingBox(alignedFace)
        if bb is None:
            print("Unable to find a face: {}".format(imgPath))
        else:
            with open(args.lp + '/' + os.path.splitext(os.path.basename(imgPath))[0] + ".json", "w") as write_file:
                json.dump(align.findLandmarks(alignedFace, bb), write_file)
        
            print("Landmarks extracted from: " + os.path.splitext(os.path.basename(imgPath))[0])

for img in args.imgs:
    getLandmarks(img)

end_time = time.time()
print('Hecho en :' + str(end_time - init_time)[:5]  +'s')