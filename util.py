import csv
import numpy as np
import pylab
from skimage import transform, filters
import matplotlib.pyplot as plt
import matplotlib.image as imm

def loadTags(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        
    tagName = [r[0] for r in data]
    return tagName, dict(zip(tagName, range(len(tagName))))

def getTagScore(scores, tags, tag2IDs):

    tagScore = []
    for r in tags:
        tagScore.append((r, scores[tag2IDs[r]]))
            
    return tagScore

def showAttMap(img, attMaps, tagName, fName, overlap = True, blur = False):
    
    
    for i in range(len(tagName)):
        attMap = attMaps[i].copy()
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'edge')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
    
        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1*(1-attMap**0.8).reshape(attMap.shape + (1,))*img + (attMap**0.8).reshape(attMap.shape+(1,)) * attMapV;
        
        imm.imsave(fName,attMap)

