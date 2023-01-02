import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import regionprops


ALT_NAMES = ["rectify2_LawrenceHoffmann","USCan_LakeSuperior","OR_JosephineCounty","pp1410b"]


def checkNeigbors(ref_color,ROI,point,search_size = (21,21),margin=10):
    '''
    Parameters
    ----------
    ref_color : list/np.array/tuple
        (R,G,B).
    ROI : np.ndarray
        ROI image.
    point : tuple/list/np.array
        (y,x).
    search_size : tuple/list, optional
        size of 2d search kernel. The default is (21,21).
    margin : int, optional
        The Maximum deviation allowed from values. The default is 10.

    Returns
    -------
    int
        Boolean to represent if neighbor with ref_color was found.

    '''
    if(type(search_size)==int):
        search_size = (search_size,search_size)
    s_y,s_x = search_size
    y_start,x_start = point
    x_start = int(max(x_start-int(s_x/2),0))
    y_start = int(max(y_start-int(s_y/2),0))
    y_end = int(min(len(ROI),x_start+s_x))
    x_end = int(min(len(ROI[0]),y_start+s_y))
    for y in range(y_start,y_end):
        for x in range(x_start,x_end):
            if(withinMargin(ref_color,ROI[y,x],margin)):
                return 1
    return 0


def superpixel(ROI):
    img = img_as_float(ROI).astype(np.float32)
    segments_slic = slic(img, n_segments=25000, compactness=30,
                         start_label=1)
    props = regionprops(segments_slic)
    return segments_slic,props


def getMedianColor(img):
    data = np.unique(np.concatenate(img),axis=0)
    if(len(data)%2==0):
        data = np.vstack([data,[data[0]]])
    return np.median(data,axis=0)

def getMasked(ROI,ref_color,margin=10,upper_thresh = 250, lower_thresh = 10):
    if((lower_thresh*3>np.sum(ref_color)) or (upper_thresh*3<np.sum(ref_color))):
        margin=1
    upper = np.array(ref_color) + margin
    lower = np.array(ref_color) - margin
    return cv2.inRange(ROI,lower,upper)
    
    
def withinMargin(ref_col,check_col,margin=10):
    r = ref_col[0]-margin<=check_col[0]<=ref_col[0]+margin
    g = ref_col[1]-margin<=check_col[1]<=ref_col[1]+margin
    b = ref_col[2]-margin<=check_col[2]<=ref_col[2]+margin
    return (r and g and b)


def backgroundClip(img, thresh = 0):
    mask = np.sum(img,axis=2)>thresh
    return img[np.ix_(mask.any(1),mask.any(0))]


def cropImageAlt(img):
    image_sample = backgroundClip(img)
    gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key = lambda x : cv2.contourArea(x),reverse = True)
    if(len(cnts)>0):
        c = max(cnts,key = lambda x : cv2.contourArea(x))
        x,y,w,h = cv2.boundingRect(c)
        crop_coords = [x,y,w,h]
        ROI = image_sample[y:y+h, x:x+w]
    else:
        crop_coords = -1
        ROI = image_sample
    return ROI,crop_coords


def cropImage(img,fname):
    fname = fname.split("/")[-1]
    if(fname in ALT_NAMES):
        return cropImageAlt(img)
    image_sample = backgroundClip(img)
    gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key = lambda x : cv2.contourArea(x),reverse = True)
    if(len(cnts)>0):
        c = max(cnts,key = lambda x : cv2.contourArea(x))
        x,y,w,h = cv2.boundingRect(c)
        crop_coords = [x,y,w,h]
        ROI = image_sample[y:y+h, x:x+w]
    else:
        crop_coords = -1
        ROI = image_sample
    return ROI,crop_coords


def getCloseCnts(point,cnts,thresh=10):
    out = []
    point = (int(point[0]),int(point[1]))
    for cnt in cnts:
        dist = np.abs(cv2.pointPolygonTest(cnt, point, True))
        if(dist<thresh):
            out.append(cnt)
    return out