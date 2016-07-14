'''
Functions to be used within Slicer to convert needle VTK Models to LabelMaps
Set UP DIR in getPaths to your own datasets directory
'''


def labelMapFromNeedle(inputVolume, needleID, needleNumber, caseNumber):
    '''
    Create a labelmap from a needle VTK model
    '''
    outputLabelMap = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.mrmlScene.AddNode(outputLabelMap)
    params = {'sampleDistance': 1, 'labelValue': needleNumber, 'InputVolume': inputVolume.GetID(),
              'surface': needleID, 'OutputVolume': outputLabelMap.GetID()}
    slicer.cli.run(slicer.modules.modeltolabelmap, None, params, wait_for_completion=True)
    slicer.util.saveNode(outputLabelMap, '/Users/guillaume/Projects/LabelMaps/%d/needle-%d.nrrd'%(caseNumber, needleNumber))
    # slicer.mrmlScene.RemoveAllObservers()
    slicer.mrmlScene.RemoveNodeReferences(outputLabelMap)
    slicer.mrmlScene.RemoveNode(outputLabelMap)
    return 0

def getPaths(DIR = '/Users/guillaume/Dropbox/GYN Cases'):
    ### Read file containing the paths to the data
    with open(DIR + '/scenes.txt') as f:
      lines = f.readlines()

    ### Create list of paths and case numbers
    paths = []
    cases = []
    for line in lines:
      paths.append(DIR + line[1:-1])
      cases.append(int(line.lstrip('./Case  ')[1:3]))
    return paths, cases


def startExtraction(start=0):
    '''
    Extract all labelmaps from needles from the case number 'start' up until the end of the dataset
    '''
    ### Get paths
    paths, cases = getPaths()
    for k, path in enumerate(paths[start:]):
        slicer.util.loadScene(path)
        nodes = slicer.util.getNodes("manual-seg*")
        backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
        slicer.util.saveNode(backgroundNode, '/Users/guillaume/Projects/LabelMaps/%d/case.nrrd'%(cases[k+start]))
        for i, node in enumerate(nodes.values()):
            labelMapFromNeedle(backgroundNode, node.GetID(), i+1, cases[k+start])
            slicer.mrmlScene.RemoveNodeReferences(node)
            # slicer.mrmlScene.RemoveObservers(node)
            slicer.mrmlScene.RemoveNode(node)
        print("start removing nodes------")
        # slicer.mrmlScene.RemoveAllObservers()
        slicer.mrmlScene.RemoveNodeReferences(backgroundNode)
        slicer.mrmlScene.RemoveNode(backgroundNode)
        # slicer.mrmlScene.Clear(0)

def extract(k):
    '''
    Extract labelmaps from needles from a single case
    '''
    ### Get paths
    paths, cases = getPaths()
    slicer.util.loadScene(paths[k])
    nodes = slicer.util.getNodes("manual-seg*")
    backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    slicer.util.saveNode(backgroundNode, '/Users/guillaume/Projects/LabelMaps/%d/case.nrrd'%(cases[k]))
    for i, node in enumerate(nodes.values()):
        labelMapFromNeedle(backgroundNode, node.GetID(), i+1, cases[k])
        # slicer.mrmlScene.RemoveNode(node)
    print("start removing nodes------")
    slicer.mrmlScene.RemoveAllObservers()
    slicer.mrmlScene.RemoveNodeReferences(backgroundNode)
    slicer.mrmlScene.RemoveNode(backgroundNode)
    # slicer.mrmlScene.Clear(0)

