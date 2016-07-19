USERPATH = os.path.expanduser("~")
print(USERPATH)
import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-k", "--k", help="case index", type=int)

args = parser.parse_args()


paths = []
needPath = []
cases = []

with open(USERPATH + '/Dropbox/GYN Cases/scenes.txt') as f:
    lines = f.readlines()

#
with open(USERPATH + '/Dropbox/GYN Cases/760CYNeedles.txt') as g:
    needles = g.readlines()

#
for line in lines:
    paths.append(USERPATH + '/Dropbox/GYN Cases' + line[1:-1])
    cases.append(int(line.lstrip('./Case  ')[1:3]))

#
for needle in needles:
    needPath.append(USERPATH + '/Dropbox/GYN Cases' + needle[1:-1])

#  
def getNeedles(case):
    return [s for s in needPath if "%03d"%int(case) in s]

#
def labelMapFromNeedle(inputVolume, needleID, value, caseNumber, name):
    '''
    Convert a needle to a labelmap, save it and remove the node from the scene.
    '''
    outputLabelMap = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.mrmlScene.AddNode(outputLabelMap)
    params = {'sampleDistance': 1, 'labelValue': value, 'InputVolume': inputVolume.GetID(),
              'surface': needleID, 'OutputVolume': outputLabelMap.GetID()}
    slicer.cli.run(slicer.modules.modeltolabelmap, None, params, wait_for_completion=True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed").SetUseLabelOutline(True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow").SetUseLabelOutline(True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed").RotateToVolumePlane(outputLabelMap)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow").RotateToVolumePlane(outputLabelMap)
    slicer.util.saveNode(outputLabelMap, USERPATH + '/Projects/LabelMaps2/%d/needle-%s.nrrd'%(caseNumber,name))
    # slicer.mrmlScene.RemoveAllObservers()
    slicer.mrmlScene.RemoveNodeReferences(outputLabelMap)
    slicer.mrmlScene.RemoveNode(outputLabelMap)
    return 0

#
def startExtractingNeedles(start=0):
    '''
    Save the needles as individual labelmaps. Read the file the scenes.txt to obtain paths.
    '''
    with open(USERPATH + '/Dropbox/GYN Cases/scenes.txt') as f:
        lines = f.readlines()
    paths = []
    cases = []
    for line in lines:
        paths.append(USERPATH + '/Dropbox/GYN Cases' + line[1:-1])
        cases.append(int(line.lstrip('./Case  ')[1:3]))
    for k, path in enumerate(paths[start:]):
        slicer.util.loadScene(path)
        nodes = slicer.util.getNodes("manual-seg*")
        backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        slicer.util.saveNode(backgroundNode, USERPATH + '/Projects/LabelMaps2/%d/case.nrrd'%(cases[k+start]))
        for i, node in enumerate(nodes.values()):
            labelMapFromNeedle(backgroundNode, node.GetID(), i+1, cases[k+start], name)
            slicer.mrmlScene.RemoveNodeReferences(node)
            # slicer.mrmlScene.RemoveObservers(node)
            slicer.mrmlScene.RemoveNode(node)
        print("start removing nodes------")
        # slicer.mrmlScene.RemoveAllObservers()
        slicer.mrmlScene.RemoveNodeReferences(backgroundNode)
        slicer.mrmlScene.RemoveNode(backgroundNode)
        # slicer.mrmlScene.Clear(0)

#
def extract(k):
    '''
    Extract needles of just one case.
    '''
    slicer.util.loadScene(paths[k])
    nodes = slicer.util.getNodes("manual-seg*")
    ndls = getNeedles(cases[k])
    backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    slicer.util.saveNode(backgroundNode, USERPATH + '/Projects/LabelMaps2/%d/case.nrrd'%(cases[k]))
    for i, ndl in enumerate(ndls):
        try:
            name = ndl.split('_')[-1].split('.')[0]
            _, node = slicer.util.loadModel(ndl, 'ndl')
            if node:
                labelMapFromNeedle(backgroundNode, node.GetID(), i+1, cases[k], name)
                slicer.mrmlScene.RemoveNode(node)
        except:
            pass
    quit()
    return 0 

extract(args.k)
