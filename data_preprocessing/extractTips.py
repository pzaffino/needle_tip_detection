'''
Old code use to extract the tips. Use as example, probably buggy
'''


from __future__ import division
import NeedleFinder
import numpy as np
import csv
import random
widget = slicer.modules.NeedleFinderWidget
l = NeedleFinder.NeedleFinderLogic()
import time as t
import os


path = [ 0 for i in range(100)]
USERPATH = os.path.expanduser("~")

list = [24,28,30,31,33,34,38,40]
#path[24] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 24 NRRD/Manual/2013-02-25-Scene-without-CtrPt.mrml'
#path[28] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 28 NRRD/Auto-Eval-LB/Scene.mrml'
#path[29] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 29 NRRD/Manual/2013-02-26-Scene-without-CtrPts.mrml'
#path[30] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 30 NRRD/Manual/2013-02-26-Scene-without-CtrPt.mrml'
#path[31] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 31 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
#path[33] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 33 NRRD/Auto-Eval-LB/2013-02-27-Scene.mrml'
#
#path[34] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 34 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
## path[35] = '/Users/guillaume/Dropbox/AMIGO Gyn Data NRRD (1)/Case 35 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
#path[37] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 37 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
#path[38] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 38 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
#path[40] = USERPATH + '/Dropbox/AMIGO Gyn Data NRRD (1)/Case 40 NRRD/Manual/2013-02-27-Scene-without-CtrPts.mrml'
path[7] = '/Users/GP1514/Projects/GYN\ Cases/Case\ \ 007/NRRD/Manual_LB/2014-01-27-Scene.mrml'
# slicer.util.loadScene(path[38])
# path = USERPATH + '/Dropbox/MachineLearning/tip.nrrd'
def cropArea(volumeNode, tipPosition, scalingConstant, filename,i):
    # TODO: draw a rectangle with the mouse
    # observer: left button pressed -> get XYZ / left button released -> get XYZ
    imageData = volumeNode.GetImageData()
    imageDimensions = imageData.GetDimensions()
    m = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(m)
    
    # create ROI
    roi = slicer.mrmlScene.CreateNodeByClass('vtkMRMLAnnotationROINode')
    slicer.mrmlScene.AddNode(roi)
    roi.SetROIAnnotationVisibility(0)
    roi.SetRadiusXYZ(10,10,10)
    roi.SetXYZ(tipPosition[0], tipPosition[1], tipPosition[2])
    roi.SetLocked(1)
    #crop volume
    cropVolumeNode =slicer.mrmlScene.CreateNodeByClass('vtkMRMLCropVolumeParametersNode')
    cropVolumeNode.SetScene(slicer.mrmlScene)
    cropVolumeNode.SetName('tip_CropVolume_node')
    cropVolumeNode.SetIsotropicResampling(False)
    slicer.mrmlScene.AddNode(cropVolumeNode)
    cropVolumeNode.SetInputVolumeNodeID(volumeNode.GetID())
    cropVolumeNode.SetROINodeID(roi.GetID())
    cropVolumeLogic = slicer.modules.cropvolume.logic()
    cropVolumeLogic.Apply(cropVolumeNode)
    # saved = slicer.util.saveNode(cropVolumeNode, filename)
    roiVolume = slicer.mrmlScene.GetNodeByID(cropVolumeNode.GetOutputVolumeNodeID())
    roiVolumeResampled = slicer.mrmlScene.AddNode(slicer.mrmlScene.CreateNodeByClass('vtkMRMLScalarVolumeNode'))
    roiVolumeResampled.SetName('resampled-'+str(i))
    parameters = {}
    parameters["InputVolume"] = roiVolume.GetID()
    parameters["OutputVolume"] = roiVolumeResampled.GetID()
    parameters["outputPixelSpacing"] = scalingConstant, scalingConstant, scalingConstant
    resamplecli = slicer.modules.resamplescalarvolume
    __cliNode = None
    __cliNode = slicer.cli.run(resamplecli, __cliNode, parameters)
    t.sleep(0.4)
    # slicer.mrmlScene.RemoveNode(roiVolume)
    return

def exportTips(filename):
    tiplist = slicer.util.getNodes('resampled-*')
    i=0
    for tip in tiplist:
        sImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( tip ) )
        saved = sitk.WriteImage( sImage, filename+"_"+str(i)+".nrrd" )
        i+=1
    # t.sleep(2)
    # slicer.mrmlScene.Clear(0)
    # slicer.mrmlScene.RemoveNode(roi)
    # roiVolume = slicer.mrmlScene.GetNodeByID(cropVolumeNode.GetOutputVolumeNodeID())
    # roiVolume.SetName("tipROI")
    return

# def resample(inputVolume, outputVolume, interpolation="linear", outputSpacing=[1.0,1.0,1.0]):
#     scene = slicer.mrmlScene
#     # inputVolume = scene.GetNodeByID(inputVolume)
#     # outputVolume = scene.GetNodeByID(outputVolume)
#     ijkToRASMatrix = vtk.vtkMatrix4x4()
#     inputVolume.GetIJKToRASMatrix(ijkToRASMatrix)
#     inputSpacing = inputVolume.GetSpacing()
#     spacingRatio = [outputSpacing[0]/inputSpacing[0], 
#                     outputSpacing[1]/inputSpacing[1], 
#                     outputSpacing[2]/inputSpacing[2]]
#     reslice = vtk.vtkImageReslice()
#     reslice.SetInputData(inputVolume.GetImageData())
#     if interpolation == "nearest neighbor":
#         reslice.SetInterpolationModeToNearestNeighbor()
#     elif interpolation == "linear":
#         reslice.SetInterpolationModeToLinear()
#     elif interpolation == "cubic":
#         reslice.SetInterpolationModeToCubic()
#     reslice.SetOutputSpacing(*spacingRatio) 
#     reslice.Update()
#     changeInformation = vtk.vtkImageChangeInformation()
#     changeInformation.SetInputData(reslice.GetOutput())
#     changeInformation.SetOutputOrigin(0.0,0.0,0.0)
#     changeInformation.SetOutputSpacing(1.0,1.0,1.0)
#     changeInformation.Update()
#     outputVolume.SetAndObserveImageData(changeInformation.GetOutput())
#     outputVolume.SetIJKToRASMatrix(ijkToRASMatrix)
#     outputVolume.SetSpacing(*outputSpacing)
#     return outputVolume

def saveCubesCropVolume(number, scalingConstant):
    slicer.util.loadScene(path[number])
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode() 
    filename = USERPATH + "/Dropbox/MachineLearning/Train_tips_cropVolume/"+str(number)
    for i in range(len(l.returnTips())):
        tipPosition = l.ijk2ras(l.returnTips()[i])
        cropArea(volumeNode, tipPosition, scalingConstant, filename, i )
    exportTips(filename)
    # slicer.mrmlScene.Clear(0)
    return "done!"

def saveRandomCubesCropVolume(number, scalingConstant, numberOfSamples=30):
    slicer.util.loadScene(path[number])
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode() 
    filename = USERPATH + "/Dropbox/MachineLearning/Random_cropVolume/"+str(number)
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    RASbounds = [0,0,0,0,0,0]
    volumeNode.GetRASBounds(RASbounds)
    xMin = RASbounds[0]
    xMax = RASbounds[1]
    yMin = RASbounds[2]
    yMax = RASbounds[3]
    zMin = RASbounds[4]
    zMax = RASbounds[5]
    edge = 10
    for i in range(numberOfSamples):
        if xMin+edge < xMax-edge:
            xR = np.random.randint(xMin+edge,xMax-edge)
        else: xR = 0
        if yMin+edge< yMax-edge:
            yR = np.random.randint(yMin+edge, yMax-edge)
        else: yR = 0
        if zMin+3 < zMax-3 : 
            zR = np.random.randint(zMin+3, zMax-3)
        else: zR = 0
        if xR*yR ==0 and xR*zR==0:
            pass
        else:
            cropArea(volumeNode, [xR,yR,zR], scalingConstant, filename, i )
    exportTips(filename)
    # slicer.mrmlScene.Clear(0)
    return "done!"




def saveCubes(number):
    slicer.util.loadScene(path[number])
    for i in range(len(l.returnTips())):
        tipPosition = l.ijk2ras(l.returnTips()[i])
        volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode() 
        sImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( volumeNode.GetName() ) )
        # volumeNode.GetIJKToRASDirectionMatrix(volumeNode)
        backgroundNode  = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        backgroundNodeName = backgroundNode.GetName()
        backgroundImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( backgroundNodeName ) )
        bgOrigin=backgroundImage.GetOrigin()
        #
        rimage = sitk.Image(40, 40, 40, sitk.sitkUInt16)
        rimage.SetSpacing([0.5,0.5,0.5])
        #rimage.SetDirection([-1,0,0,0,-1,0,0,0,1])
        rimage.SetOrigin([-tipPosition[0]-10,-tipPosition[1]-10,tipPosition[2]-10]) #-np.array([1,1,1]
        #rimage.SetDirection([1,0,0]
        tx = sitk.Transform()
        #
        lilImage = sitk.Resample(sImage, rimage, tx, sitk.sitkNearestNeighbor, sitk.sitkUInt16)
        newNode=slicer.mrmlScene.CreateNodeByClass('vtkMRMLScalarVolumeNode')
        slicer.mrmlScene.AddNode(newNode)
        newNode.SetName('cuby')
        sitk.WriteImage( lilImage, sitkUtils.GetSlicerITKReadWriteAddress( newNode.GetName() ) )
        lilImage.SetOrigin([0,0,0])
        filename = USERPATH + "/Projects/Train_tips/"+str(number)+ "-"+str(i)+".nrrd"
        # print filename
        sitk.WriteImage( lilImage, filename )


def saveCubesShifted(number, numberOfSamples=30):
    slicer.util.loadScene(path[number])
    for i in range(len(l.returnTips())):
        for j in range(numberOfSamples):
            a = np.random.randint(-2,2)
            b = np.random.randint(-2,2)
            c = np.random.randint(-2,2)
            tipPosition = l.ijk2ras(l.returnTips()[i])
            volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
            sImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( volumeNode.GetName() ) )
            # volumeNode.GetIJKToRASDirectionMatrix(volumeNode)
            backgroundNode  = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
            backgroundNodeName = backgroundNode.GetName()
            backgroundImage = sitk.ReadImage( sitkUtils.GetSlicerITKReadWriteAddress( backgroundNodeName ) )
            bgOrigin=backgroundImage.GetOrigin()
            #
            rimage = sitk.Image(40, 40, 40, sitk.sitkUInt16)
            rimage.SetSpacing([0.5,0.5,0.5])
            #rimage.SetDirection([-1,0,0,0,-1,0,0,0,1])
            rimage.SetOrigin([-tipPosition[0]-10 + a,-tipPosition[1]-10 + b,tipPosition[2]-10 + c ]) #-np.array([1,1,1]
            #rimage.SetDirection([1,0,0]
            tx = sitk.Transform()
            #
            lilImage = sitk.Resample(sImage, rimage, tx, sitk.sitkNearestNeighbor, sitk.sitkUInt16)
            newNode=slicer.mrmlScene.CreateNodeByClass('vtkMRMLScalarVolumeNode')
            slicer.mrmlScene.AddNode(newNode)
            newNode.SetName('cuby')
            sitk.WriteImage( lilImage, sitkUtils.GetSlicerITKReadWriteAddress( newNode.GetName() ) )
            lilImage.SetOrigin([0,0,0])
            filename = USERPATH + "/Dropbox/MachineLearning/Train_tips_shifted/"+ str(number)+ "_" +str(i)+"-"+ str(j)+ "_" + str(np.random.randint(9999999)) + "-"+str(i+j)+".nrrd"
            # print filename
            sitk.WriteImage( lilImage, filename )


def saveRandomCubes(number, numberOfSamples=30):
    slicer.util.loadScene(path[number])
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
        filename = USERPATH + "/Dropbox/NeedleFinderProjectWeek/NeedleTipsRandomNRRD/"+ str(np.random.randint(9999999)) + "-random_"+str(i)+".nrrd"
        # print filename
        sitk.WriteImage( lilImage, filename )
