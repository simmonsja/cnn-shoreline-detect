import numpy as np
import matplotlib.pyplot as plt

from .data_preprocessing import mask_to_uv
import torchvision.utils as tutils

import imageio

import cv2

################################################################################
################################################################################

def torch2plt(img):
    # extract numoy array and reshape for plotting
    npimg = img.numpy()
    return np.transpose(npimg, (1,2,0))

################################################################################
################################################################################

def mask2binary(mask, thres):
    # thresholding
    mask = mask.numpy().squeeze()
    outMask = np.zeros(mask.shape)
    outMask[mask>thres] = 1
    return outMask

################################################################################
################################################################################

def plot_predictions(prntNum, dataX, dataY, dataPred, jj, thres):
    '''
    Plot the raw predicitons that allow you to view the activations through the
    layers.
    '''
    # plot up the base data
    dataX, dataY, dataPred = dataX[prntNum], dataY[prntNum], dataPred[prntNum]
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(torch2plt(dataX))
    ax1.scatter(mask_to_uv(dataY[0,...])[0],mask_to_uv(dataY[0,...])[1],s=5,color='r',alpha=0.5)
    predMask = mask2binary(dataPred[jj],thres)
    ax1.scatter(mask_to_uv(predMask)[0],mask_to_uv(predMask)[1],s=5,color='b',alpha=0.05)

    # now plot the individual layers
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(torch2plt(tutils.make_grid(dataPred, nrow=3, padding=0)))
    _, xMax = ax1.get_xlim()
    yMin, _ = ax1.get_ylim()
    jj = 0
    ii = 0
    for prntNum, _ in enumerate(dataPred):
        ax1.text(ii*xMax/3 + xMax/50, jj * yMin/2 + yMin/15, prntNum.__str__(),fontdict={'color':'r','size':18,'weight':'bold'})
        if ii == 2:
            jj += 1
            ii = 0
        else:
            ii += 1

################################################################################
################################################################################

def plot_refined_predictions(prntNum, dataX, dataY, dataPred, thres,cvClean=False,imReturn=False):
    '''
    Plot the refined and final predicitons through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # plot up the base data
    dataX, dataY, dataPred = dataX[prntNum], dataY[prntNum], dataPred[prntNum]
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(torch2plt(dataX))
    ax1.scatter(mask_to_uv(dataY[0,...])[0],mask_to_uv(dataY[0,...])[1],s=5,color='r',alpha=0.5)

    # # would be nice to implement some cleaning of the data to get a crisper shoreline

    # for debugging
    # plot_refined_predictions(10,
    #                      dataX=imData,
    #                      dataY=np.zeros(imData.shape),
    #                      dataPred=model_pred,
    #                      thres=0.8,
    #                      imReturn=False)

    if cvClean:
        combined = dataPred[1]* 0.75 + dataPred[2] * 0.25
        cvIm = (combined.numpy().squeeze()*255).astype(np.uint8)

        # find the shoreline blobs as contours
        ret,thresh = cv2.threshold(cvIm,int(thres*255),255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # can we find some feature that distinguishes a shoreline?
        isConvex = np.array([cv2.isContourConvex(_) for _ in contours])
        area = np.array([cv2.contourArea(_) for _ in contours])
        perimeter = np.array([cv2.arcLength(_,True) for _ in contours])
        radius = []
        for _ in contours:
            _,tmpRadius = cv2.minEnclosingCircle(_)
            radius.append(tmpRadius)
        # Settle for just cutting down some noise at the moment
        contInd = np.where((np.array(radius) > 0.25*np.array(radius).max()) & (perimeter > 0.25*perimeter.max()))[0]
        #contInd = np.where((perimeter > np.quantile(perimeter,0.9)) & (area > np.quantile(area,0.9)))[0]
        if contInd.shape[0] < 1:
            contInd = np.array([np.array([_.shape[0] for _ in contours]).argmax()])
        #ax1.scatter(contUV[:,0],contUV[:,1],s=5,color='m',alpha=0.7)

        predMask = mask2binary(combined,thres)
        predU = mask_to_uv(predMask)[0]
        predV = mask_to_uv(predMask)[1]
        contBool = np.full((predU.shape[0],),False)
        for ii, (thisU, thisV) in enumerate(zip(predU,predV)):
            thisBools = []
            for _ in contInd:
                thisBools.append(cv2.pointPolygonTest(contours[_],(thisU,thisV),True) > 0)
            contBool[ii] = np.any(thisBools)
        ax1.scatter(predU[contBool],predV[contBool],s=5,color='m',alpha=0.07)
    else:
        # # perform a weighted combination
        combined = dataPred[1]* 0.5 + dataPred[2] * 0.5
        predMask = mask2binary(combined,thres)
        ax1.scatter(mask_to_uv(predMask)[0],mask_to_uv(predMask)[1],s=5,color='m',alpha=0.07)

    if imReturn:
        # this is for writing a gif output
        ax1.axis('off')
        fig.canvas.draw()
        imData = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.canvas.get_width_height()
        midW = int(np.round(width/2))
        midH = int(np.round(height/2))
        imData = imData.reshape((width, height, 3))
        startH = np.where(imData[midW,:,0]<255)[0][0]
        endH = np.where(imData[midW,:,0]<255)[0][-1]
        startW = np.where(imData[:,midH,0]<255)[0][0]
        endW = np.where(imData[:,midH,0]<255)[0][-1]
        imData = imData[startW:endW,startH:endH,:]
        plt.close()
        return imData

################################################################################
################################################################################

def write_output_gif(gifName,dataX, dataY, dataPred, thres,cvClean=False):
    # write the final prediction into a gif for visualisation
    imageio.mimsave(gifName,
                    [plot_refined_predictions(_,dataX, dataY, dataPred, thres,cvClean,imReturn=True) for _ in np.arange(dataX.shape[0])],
                    fps=0.5)
