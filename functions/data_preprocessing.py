import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import shutil

import torch
from torch.utils import data
from torchvision.utils import save_image
import skimage

import js_general as jsg

################################################################################
################################################################################

def convert_to_scale(pntU, pntV, scaleDict):
    # where:
    # pntU are the coords in width
    # pntV are the coords in height
    if scaleDict['cropHWBool']:
        # crop height
        outU = pntU / scaleDict['scaleConvert']
        outV = (pntV - scaleDict['cropConvert'] ) / scaleDict['scaleConvert']
    else:
        # crop width
        outV = pntV / scaleDict['scaleConvert']
        outU = (pntU - scaleDict['cropConvert']) / scaleDict['scaleConvert']
    return outU, outV

################################################################################
################################################################################

def get_scale_info(inScale, targetSize):
    if abs(inScale[0]/inScale[1] - targetSize[0]/targetSize[1]) > 0.01:
        cropBool = True
        if inScale[0]/inScale[1] > targetSize[0]/targetSize[1]:
            # crop height
            cropHWBool = True
            cropHW = np.array((targetSize[0]/targetSize[1]) * inScale[1]).round().astype(int)
            cropConvert = ((inScale[0] - cropHW) / 2).round().astype(int)
            scaleConvert = inScale[1] / targetSize[1]
        else:
            # crop width
            cropHWBool = False
            cropHW = np.array((targetSize[1]/targetSize[0]) * inScale[0]).round().astype(int)
            cropConvert = ((inScale[1] - cropHW) / 2).round().astype(int)
            scaleConvert = inScale[0] / targetSize[0]
    else:
        cropBool = False
        cropHW = None
        cropHWBool = None
        cropConvert = None
        scaleConvert = None

    dictOut = {
        'cropBool': cropBool,
        'scaleConvert': scaleConvert,
        'cropHWBool': cropHWBool,
        'cropHW': cropHW,
        'cropConvert': cropConvert,
    }

    return dictOut

################################################################################
################################################################################

def augment_images_kp(dataX, dataY, seq):
    # Augment images with keypoints
    # dataX = (n, w, h)

    if dataX.max() <= 1:
        dataX = dataX * 255
        dataX = dataX.round().astype(np.uint8)

    # map train_Y to keypoint_Y
    segmap = [np.vstack(mask_to_uv(_)).T for _ in dataY]

    # do the actual augmentation
    print('Performing augmentation...')
    aug_X, aug_kp_Y = seq(images=dataX,keypoints=segmap)

    #transform Y back to mask
    print('Transforming Y to mask...')
    aug_Y = np.zeros(dataY.shape)
    for ii, thisKp in enumerate(aug_kp_Y):
        jsg.progress_bar(ii,aug_kp_Y.__len__())
        thisY = uv_to_mask(thisKp[:,0],thisKp[:,1],(dataX.shape[1],dataX.shape[2]))
        aug_Y[ii,...] = thisY

    #transform back to normalised
    aug_X = aug_X / 255
    print('\nDone!')
    return aug_X, aug_Y


################################################################################
################################################################################

def load_images(imagePaths,targetSize,edgeCoords=False):
    # loads in the images in the list imagePaths and transforms them to the
    # correct size if edgeCoords are provided (as list of U,V arrays the same
    # length as imageNames) then these will be transformed with the image

    allImData = np.full((imagePaths.__len__(),targetSize[0],targetSize[1],3),np.nan)
    allUVData = np.full((imagePaths.__len__(),targetSize[0],targetSize[1]),np.nan)

    if not edgeCoords:
        UVData = [None] * imagePaths.__len__()
        uvBool = False
    else:
        UVData = edgeCoords
        uvBool = True

    #get image size:
    infoI = cv2.imread(imagePaths[0])
    cropInfo = get_scale_info(infoI.shape,targetSize)

    print('Loading {} images...'.format(imagePaths.__len__()))
    for ii, (imPath, thisUV) in enumerate(zip(imagePaths,UVData)):
        jsg.progress_bar(ii,imagePaths.__len__())

        #read the image and convert to RGB
        origI = cv2.imread(imPath)
        origI = cv2.cvtColor(origI, cv2.COLOR_BGR2RGB)

        if origI.shape[0] > origI.shape[1]:
            # if the shape is off then lets just rotate and keep trucking
            origI = skimage.transform.rotate(origI,-90,resize=True)

        if uvBool:
            # shoreline mask
            shlMask = np.zeros(origI.shape[0:2])
            for _ in thisUV.astype(int):
                if _[1] < origI.shape[0] and _[0] < origI.shape[1]:
                    shlMask[_[1],_[0]] = 1
            newShlMask = np.zeros(targetSize)
            newU, newV = convert_to_scale(thisUV[:,0],thisUV[:,1],cropInfo)
            # NEED to check the UV coords dont magically stray to negative or too big numbers
            for _ in zip(newU, newV):
                if _[1] < targetSize[0] and _[0] < targetSize[1] and _[0] > 0 and _[1] > 0:
                    newShlMask[int(_[1]),int(_[0])] = 1

        if cropInfo['cropBool']:
            # crop before resize
            if cropInfo['cropHWBool']:
                cropDiff = int((origI.shape[0]-cropInfo['cropHW'])/2)
                cropI = origI[cropInfo['cropConvert']:-cropInfo['cropConvert'],...]
            else:
                cropDiff = int((origI.shape[1]-cropInfo['cropHW'])/2)
                cropI = origI[:,cropInfo['cropConvert']:-cropInfo['cropConvert'],:]
            origI = cropI.copy()

        # now resize
        newI = cv2.resize(origI,targetSize[::-1])

        if newI.max() > 1:
            newI = newI / 255

        # and assign the data back to an array
        # OR could save this data to an image?
        allImData[ii,...] = newI.copy()
        if uvBool:
            allUVData[ii,...] = newShlMask.copy()

    print('Done!')
    return allImData, allUVData

################################################################################
################################################################################

def mask_to_uv(mask):
    # output pntU (width) and pntV (height)
    pntU = np.where(mask)[1] + 0.5
    pntV = np.where(mask)[0] + 0.5
    return pntU, pntV

################################################################################
################################################################################

def uv_to_mask(pntU,pntV,targetSize):
    # input numpy arrays of pntU (width) and pntV (height)
    # output mask
    mask = np.zeros(targetSize)
    pntU = np.floor(pntU).astype(int)
    pntV = np.floor(pntV).astype(int)
    for _ in zip(pntU, pntV):
        if _[1] < targetSize[0] and _[0] < targetSize[1] and _[0] > 0 and _[1] > 0:
            mask[_[1],_[0]] = 1
    return mask


################################################################################
################################################################################

def image_to_255(im):
    im = im * 255
    return im.round().astype(int)

################################################################################
################################################################################

def cv_to_torch(im):
    # use numpy move axis to get channels first
    im = np.moveaxis(im,2,0)
    return im
################################################################################
################################################################################

def load_train_test_imagedata(basePath):
    # Loading the objects:
    with open(os.path.join(basePath,'labelData.pkl'), 'rb') as f:
        partition, labels = pickle.load(f)
    return partition, labels

################################################################################
################################################################################

def save_train_test_imagedata(basePath, trainX, trainY, testX, testY):
    outFormat = 'id_{}'

    labels = {}

    #refresh
    os.makedirs(os.path.dirname(basePath), exist_ok=True)
    shutil.rmtree(basePath)
    os.makedirs(basePath)

    print('Loading {} training images...'.format(trainX.shape[0]))
    trainLabs = []
    idNum = 1
    for ii, (im, lab) in enumerate(zip(trainX, trainY)):
        jsg.progress_bar(ii,trainX.shape[0])
        imName = outFormat.format(idNum)
        save_image(torch.from_numpy(cv_to_torch(im)),os.path.join(basePath,imName+'.jpg'))
        trainLabs.append(imName)
        labels[imName] = np.expand_dims(lab,0) # torch format
        idNum += 1

    print('\nLoading {} test images...'.format(testX.shape[0]))
    testLabs = []
    for ii, (im, lab) in enumerate(zip(testX, testY)):
        jsg.progress_bar(ii,testX.shape[0])
        imName = outFormat.format(idNum)
        save_image(torch.from_numpy(cv_to_torch(im)),os.path.join(basePath,imName+'.jpg'))
        testLabs.append(imName)
        labels[imName] = np.expand_dims(lab,0) # torch format
        idNum += 1

    partition = {
        'train': trainLabs,
        'validation': testLabs,
    }

    # Saving the objects:
    with open(os.path.join(basePath,'labelData.pkl'), 'wb') as f:
        pickle.dump([partition, labels], f)
    return partition, labels

################################################################################
################################################################################

class torchLoader(data.Dataset):
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    #'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, basePath):
        #'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.basePath = basePath

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #print(os.path.join(self.basePath,'{}.jpg'.format(ID)))
        imIn = skimage.io.imread(os.path.join(self.basePath,'{}.jpg'.format(ID)))
        imIn = imIn / 255
        X = torch.from_numpy(imIn.transpose((2, 0, 1))).float()
        #X = torch.load(os.path.join(self.basePath,'{}.jpg'.format(ID)))
        y = torch.from_numpy(self.labels[ID]).float()

        return X, y

################################################################################
################################################################################
