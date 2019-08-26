# import libraries
import math
import os, time
import numpy as np
from PIL import Image
import os.path as osp
import copy

# import torch modules
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils as tutils
import torchvision
from torch.autograd import Variable

import js_general as jsg
from .data_preprocessing import torchLoader

################################################################################
################################################################################

# There are plenty of HED implementations (in pytorch/keras/others) out there on
# Github including the original authors (https://github.com/s9xie/hed) in caffe.
# I am not going to re-write perfectly good code so I will base my code on
# the implentation of https://github.com/buntyke/pytorch-hed which I believe in
# turn may borrow from others.

# Functions:
#   - pretrained_weights
#   - weights_init
#   - transfer_weights
#   - convert_vgg

# and classes:
#   - vgg
#   - hed_cnn
#     - trainer (almost completely modified)

# are taken with small modifications.

################################################################################
################################################################################
# Misc. functions
################################################################################
################################################################################

def convert_vgg(vgg16):
    # convert vgg pretrained weights to HED weights
    net = vgg()
    vgg_items = list(net.state_dict().items())
    vgg16_items = list(vgg16.items())
    pretrain_model = {}
    j = 0
    for k, v in net.state_dict().items():
        v = vgg16_items[j][1]
        k = vgg_items[j][0]
        pretrain_model[k] = v
        j += 1
    return pretrain_model

############################################################################
############################################################################

def hed_predict(model, images, bSize=4):
    '''
    This gives the final predictions out (with sigmoid activation)
    '''
    allout = torch.zeros((images.shape[0],6,1,images.shape[2],images.shape[3]))
    for ii in np.arange(0, images.shape[0], bSize):
        d1, d2, d3, d4, d5, d6 = model.forward(images[ii:ii+bSize,...])

        # transform with sigmoid function
        sigFun = nn.Sigmoid()
        d1 = sigFun(d1)
        d2 = sigFun(d2)
        d3 = sigFun(d3)
        d4 = sigFun(d4)
        d5 = sigFun(d5)
        d6 = sigFun(d6)

        allout[ii:ii+bSize,...] = torch.cat([d1,d2,d3,d4,d5,d6],1).unsqueeze(2)
    return allout.detach()

################################################################################
################################################################################

def pretrained_weights(model, weightsPath=None, applyWeights=False, hedIn=False):
    # convert vgg pretrained weights to HED weights
    if applyWeights:
        model.apply(weights_init)

        #grab a pretrained vgg
        pretrained_dict = torch.load(weightsPath, map_location='cpu')
        if not hedIn:
            pretrained_dict = convert_vgg(pretrained_dict)

            #now transfer onto my old model
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
        else:
            model_dict = pretrained_dict
        model.load_state_dict(model_dict)
    return model
################################################################################
################################################################################

def transfer_weights(model_from, model_to):
    # Transfer the vgg weights to the new HED model
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)

################################################################################
################################################################################

def weights_init(m):
    # if training from scratch we can initialise with nice weights
    classname = m.__class__.__name__
    #print (classname)
    if classname.find('Conv2d') != -1:
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)

################################################################################
################################################################################
# Model classes
################################################################################
################################################################################

class vgg(nn.Module):
    '''
    Define the vgg model which we need for the pretrained weights
    '''
    def __init__(self):
        super(vgg, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=35),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    ############################################################################
    ############################################################################

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5

################################################################################
################################################################################

class hed_cnn(nn.Module):
    '''
    And now define the cnn model which is slightly different from the vgg model
    with outputs at various points in the model.
    '''
    def __init__(self):
        super(hed_cnn, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    ############################################################################
    ############################################################################

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # output from various layers within the model (various levels of
        # abstraction).
        d1 = self.dsn1(conv1)
        d2 = F.upsample(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample(self.dsn4(conv4), size=(h,w))
        d5 = F.upsample(self.dsn5(conv5), size=(h,w))

        # by combining these (a learned combination) we can get the best estimate
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        # d1 = F.sigmoid(d1)
        # d2 = F.sigmoid(d2)
        # d3 = F.sigmoid(d3)
        # d4 = F.sigmoid(d4)
        # d5 = F.sigmoid(d5)
        # fuse = F.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse

################################################################################
################################################################################
# Trainer class
################################################################################
################################################################################

class Trainer(object):
    # init function for class
    def __init__(self, model, partition, labels, params):
        # model
        self.model = model

        # hardware settings
        self.cuda = params['cuda']
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        self.threshold = 0.5 # for binary

		# some weighting and decay for regularisation
        self.regWeight = 5.
        #bce weight decay
        self.decayCntr = 0
        self.regDecay = 0.2
        self.regEpochs = [10,12,15,20,22,26,27,30,32,34,36,38]

		# which loss function to use - binary cross entropy weighted or weighted with regularisation
        if params['lossFunction'] == 'weightedBCEReg':
            self.lossFn = self.weighted_bce_regul
            self.regDecay = 1
        else:
            self.lossFn = self.weighted_bce
        # optimiser
        self.optimiser = params['optimiser']

		# training generator parameters
        tgParams = {
            'batch_size': params['batchSize'],
            'shuffle': True,
            'num_workers': 6,
        }

        # training and data loading settings
        self.trainingSet = torchLoader(partition['train'], labels, params['basePath'])
        self.trainingGenerator = tutils.data.DataLoader(self.trainingSet, **tgParams)

        self.validationSet = torchLoader(partition['validation'], labels, params['basePath'])
        self.validationGenerator = tutils.data.DataLoader(self.validationSet, **tgParams)

        # general training and leaarning params
        self.epochs = params['epochs']
        self.batchSize = params['batchSize']

        self.reduceLRBool = False
        self.lrDecay = params['lrDecay']

############################################################################
############################################################################
############################################################################
############################################################################
# Training/prediction functions

    def train(self):
        if self.cuda:
            self.model.cuda()
        self.model.train()

		# store best weight to restore at the end
        best_model_wts = copy.deepcopy(self.model.state_dict())
        bestAcc = np.inf
        bestLoss = np.inf

        lossHistory = []
        for epoch in range(self.epochs):
            print('Epoch {}/{}:'.format(epoch+1, self.epochs))
            # initialize gradients
            self.optimiser.zero_grad()

            # adjust hed learning rate
            if not (epoch+1) % 10 and self.reduceLRBool:
                self.adjustLR()

            sumLoss = 0.0

            # train the network
            num = 0
            for local_batch, local_labels in self.trainingGenerator:
                jsg.progress_bar(num,self.trainingGenerator.__len__())
                num += 1
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):
                    # predict and get loss
                    d1, d2, d3, d4, d5, d6 = self.model(local_batch)

                    # calc loss
                    loss1 = self.lossFn(d1, local_labels)
                    loss2 = self.lossFn(d2, local_labels)
                    loss3 = self.lossFn(d3, local_labels)
                    loss4 = self.lossFn(d4, local_labels)
                    loss5 = self.lossFn(d5, local_labels)
                    loss6 = self.lossFn(d6, local_labels)

                    # add all losses with equal weight for training
                    loss = loss1 + loss2 + loss3 + loss4 +  loss5 + loss6

                    # backward + optimize
                    backLoss = loss.sum()
                    backLoss.backward()
                    self.optimiser.step()

                    sumLoss += backLoss/self.batchSize
            # perform validation every epoch (maybe overkill)
            valScore = self.validation(epoch+1)

            # deep copy the model to take the best at the end
            if valScore < bestAcc:
                bestAcc = valScore
                best_model_wts = copy.deepcopy(self.model.state_dict())

            if sumLoss < bestLoss:
                bestLoss = sumLoss.detach()

            if epoch + 1 in self.regEpochs:
                self.adjustRegWeight()

            # print loss
            print('Epoch: {}; Loss: {:.4f}; Val Loss {:.4f}'.format(epoch+1, sumLoss, valScore))
            lossHistory.append(sumLoss)
            sumLoss = 0.0

            # save model after every epoch (could be useful)?
            # torch.save()
        # restore the best performing model
        print('Restoring the best performing model...')
        self.model.load_state_dict(best_model_wts)
        print('Done...')
        return lossHistory

    ############################################################################
    ############################################################################

    def validation(self, epoch):
        # eval model on validation set
        self.model.eval()
        lossAcc = 0
        for local_batch, local_labels in self.validationGenerator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

            d1, d2, d3, d4, d5, d6 = self.model(local_batch)
            with torch.set_grad_enabled(False):
                # calc loss
                loss1 = self.lossFn(d1, local_labels, weight=5)
                loss2 = self.lossFn(d2, local_labels, weight=5)
                loss3 = self.lossFn(d3, local_labels, weight=5)
                loss4 = self.lossFn(d4, local_labels, weight=5)
                loss5 = self.lossFn(d5, local_labels, weight=5)
                loss6 = self.lossFn(d6, local_labels, weight=5)

                # add all losses with equal weight
                # loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                # we will optimise for the two we use.
                loss = loss2 + loss3

            lossAcc += loss.item() / local_batch.shape[0]
        self.model.train()
        return lossAcc

    ############################################################################
    ############################################################################


    def predict(self, trainval='validation'):
        '''
        Easy predition after training a model (from trainer class). hed_predict
        can be used if just predicting on unseen data.
        '''
        with torch.set_grad_enabled(False):
            # perform forward computation
            if trainval == 'validation':
                # local_batch, local_labels = [_[0:1] for _ in iter(self.validationGenerator).next()]
                local_batch, local_labels = iter(self.validationGenerator).next()
            else:
                # local_batch, local_labels = [_[0:1] for _ in iter(self.trainingGenerator).next()]
                local_batch, local_labels = iter(self.trainingGenerator).next()

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

            d1, d2, d3, d4, d5, d6 = self.model.forward(local_batch)

            # transform with sigmoid activation
            sigFun = nn.Sigmoid()
            d1 = sigFun(d1)
            d2 = sigFun(d2)
            d3 = sigFun(d3)
            d4 = sigFun(d4)
            d5 = sigFun(d5)
            d6 = sigFun(d6)

            allout = torch.cat([d1,d2,d3,d4,d5,d6],1).unsqueeze(2)
        return local_batch.cpu(), allout.cpu(), local_labels.cpu().numpy()

    def _assertNoGrad(self, variable):
        # From https://github.com/buntyke/pytorch-hed
        assert not variable.requires_grad

    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    # Loss Functions

    def weighted_bce(self, output, target, weight=10):
        # Plain binary cross entropy loss implemented using pytorch BCEWithLogitsLoss
        totalNum = target.cpu().numpy().size
        posNum = target.cpu().numpy().sum()
        posWeight = (totalNum - posNum) / posNum
        # to debug
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(posWeight)),reduction='None')
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(posWeight)),reduction='mean')
        loss = criterion(output, target)
        return loss

    ############################################################################
    ############################################################################

    def weighted_bce_regul(self, output, target, weight=10):
        # Binary cross entropy loss with additional regularisation to bring minimise
        # the number of elements and create a more crisp shoreline prediction.
        if not weight:
            weight = self.regWeight
        totalNum = target.cpu().numpy().size
        posNum = target.cpu().numpy().sum()
        posWeight = (totalNum - posNum) / posNum
        # to debug
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(posWeight)),reduction='None')
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(posWeight)),reduction='mean')
        loss = criterion(output, target)
        # add regularisation
        loss += ((output>0.5).sum() - target.sum()).abs() * weight / output.nelement()
        return loss

    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    # Utility Functions

    def adjustLR(self):
        # From https://github.com/buntyke/pytorch-hed
        # utility function to adjust the learning rate
        for param_group in self.optimiser.param_groups:
            param_group['lr'] *= self.lrDecay

    ############################################################################
    ############################################################################

    def adjustRegWeight(self):
        # loss contribution of bce
        self.regWeight *= self.regDecay
