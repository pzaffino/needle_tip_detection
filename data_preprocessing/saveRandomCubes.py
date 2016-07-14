'''
Code to extract random cubes from the MRI
TODO:   - create a parameter for the shape of the data
        - create a parameter for the spacing 
'''
# save random cubes
from __future__ import division
import NeedleFinder
import numpy as np
import csv
import random
import time as t
import os

def startExtracting(start=0):
	with open('/Users/guillaume/Dropbox/GYN Cases/scenes.txt') as f:
		lines = f.readlines()
	paths = []
	cases = []
	for line in lines:
		paths.append('/Users/guillaume/Dropbox/GYN Cases' + line[1:-1])
		cases.append(int(line.lstrip('./Case  ')[1:3]))
	for k, path in enumerate(paths[start:]):
		saveRandomCubes(path, cases[k], 30)


def saveRandomCubes(path, number, numberOfSamples=30):
    slicer.util.loadScene(path)
    for i in range(numberOfSamples):
        volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        RASbounds = [0,0,0,0,0,0]
        volumeNode.GetRASBounds(RASbounds)
        xMin = RASbounds[0]
        xMax = RASbounds[1]
        yMin = RASbounds[2]
        yMax = RASbounds[3]
        zMin = RASbounds[4]
        zMax = RASbounds[5]
        xR = np.random.randint(xMin+15,xMax-15)
        yR = np.random.randint(yMin+15, yMax-15)
        zR = np.random.randint(zMin+15, zMax-15)
        sImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( volumeNode.GetName() ) )
        # volumeNode.GetIJKToRASDirectionMatrix(volumeNode)
        backgroundNode  = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        backgroundNodeName = backgroundNode.GetName()
        backgroundImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( backgroundNodeName ) )
        bgOrigin=backgroundImage.GetOrigin()
        #
        rimage = sitk.Image(20,20,20, sitk.sitkUInt16)
        rimage.SetSpacing([1,1,1])
        #rimage.SetDirection([-1,0,0,0,-1,0,0,0,1])
        rimage.SetOrigin([-xR-10,-yR-10,zR-10]) #-np.array([1,1,1]
        #rimage.SetDirection([1,0,0]
        tx = sitk.Transform()
        #
        lilImage = sitk.Resample(sImage, rimage, tx, sitk.sitkNearestNeighbor, sitk.sitkFloat32)
        newNode=slicer.mrmlScene.CreateNodeByClass('vtkMRMLScalarVolumeNode')
        slicer.mrmlScene.AddNode(newNode)
        newNode.SetName('cuby')
        sitk.WriteImage( lilImage, sitkUtils.GetSlicerITKReadWriteAddress( newNode.GetName() ) )
        lilImage.SetOrigin([0,0,0])
        filename = USERPATH + "/Dropbox/2016-projectweekdata/RandomCubes20-1/"+ str(np.random.randint(9999999)) + "-random_"+str(i)+".nrrd"
        # print filename
        sitk.WriteImage( lilImage, filename )
        slicer.mrmlScene.Clear(0)