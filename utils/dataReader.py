import os
import json


def getNames(dataPath):
    dataPaths = os.listdir(dataPath)
    return [dataPath + i.split(".")[0] for i in dataPaths if (".json" in i)]


def loadJson(name):
    fname = name+".json"
    with open(fname,"r") as f:
        json_data = json.load(f)
    return json_data
    
def getTemplateCoords(shape):
    points = shape["points"]
    (x_min,y_min),(x_max,y_max) = points
    if(x_max<x_min):
        x_max,x_min = x_min,x_max
    if(y_max<y_min):
        y_max,y_min = y_min,y_max
        
    return y_min,y_max,x_min,x_max
