import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, width, height, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[0:width:1, 
                      0:height:1]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    zz = np.exp(kde_skl.score_samples(xy_sample))
    zz = np.reshape(zz,xx.shape)


    cz = np.exp(kde_skl.score_samples(np.array([[width//2,height//2]]))).item()
    cz = (cz-np.average(zz))/np.std(zz)

    return xx, yy,zz,cz

def calculate(image,width=400,height=400,threshold=20,iterations=0):
    
    color_list = [(68,58,130),(49,104,142),(33,144,141),(53,183,121),(143,215,68),(253,231,37)]
    level_list = [3,7,16,25,40,60]
    image = cv2.resize(image,(width,height))

    footprints,areas,heights,center_points = [],[],[],[]
    for i in range(len(color_list)):
        r,g,b = color_list[i]
        level = level_list[i]
        color = np.uint8([int(b),int(g),int(r)])
        low_b = color-threshold
        up_b = color+threshold
        map = cv2.inRange(image,low_b,up_b) 
        _ , thresh = cv2.threshold(map,127,255,cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=iterations)
        contours, hierarchy = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            footprint = cv2.contourArea(c)
            if footprint < 50:
                continue
            area = footprint*level

            rect = cv2.minAreaRect(c)
            center_points.append(rect[0])

            footprints.append(footprint) 
            areas.append(area) 
            heights.append(level*3)

    center_points = np.array(center_points)
    x = center_points[:,0]
    y = center_points[:,1]
    xx, yy, zz, cz = kde2D(x, y, 50,width,height)

    BCR = np.sum(footprints)/(width*height)
    FAR = np.sum(areas)/(width*height)
    KD = cz
    data = (BCR,FAR,KD)

    return data,contours,heights


if __name__ == '__main__':
    
    dir = r'datasets\shenzhen\train\1'
    file_list = [os.path.join(dir,x) for x in os.listdir(dir) if x[-3:]=='png']
    df = pd.DataFrame(columns=['FILE','BCR','FAR','KD'])
   
    for i  in tqdm(range(len(file_list))):
        FILE = file_list[i]
        image = cv2.imread(FILE)
        data,contours,heights = calculate(image)
        BCR,FAR,KD = data
        df.loc[i] =  dict(zip(df.columns, [FILE,BCR,FAR,KD]))
    
    df.to_excel('analysis/train_data2.xlsx',index=True)
    
