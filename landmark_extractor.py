#!/usr/bin/env python2

import json

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)
import openface

fileDir = '/root/openface'
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
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
    with open('./landmarks/' + os.path.splitext(os.path.basename(imgPath))[0] + ".json", "w") as write_file:
        json.dump(align.findLandmarks(rgbImg, bb), write_file)
    
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))

    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))


    bb = align.getLargestFaceBoundingBox(alignedFace)

for img in args.imgs:
    getLandmarks(img)
