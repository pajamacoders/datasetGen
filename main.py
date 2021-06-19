import os
import numpy as np
import h5py
import pdb
import pandas as pd
import cv2
import torch
import time
import ast
if __name__=='__main__':
    root = '/home/sabi/workspace/ganhackerton/dataset'
    with os.scandir(root) as entries:
        fpathes = [os.path.join(root, fname) for fname in entries if os.path.isfile(fname)]
    dfall = pd.concat([pd.read_csv(fpath) for fpath in fpathes[:1]]).drop('Unnamed: 0',axis=1)
    basename = '/main/dataset/'
    pts = np.zeros((1000,2))
    tmp = np.zeros((512,512),dtype=np.uint8)
    start = time.time()
    for j in range(10):
        for i in range(2000):
            path, poly = dfall.iloc[i]['path'].replace(basename, root), dfall.iloc[i]['poly']
            img = cv2.imread(path)
            poly = ast.literal_eval(poly)
            tmp.fill(0)
            for pl in poly:
                j=0
                for p in pl:
                    x,y = p['x'], p['y']
                    pts[j] = x,y
                    j+=1

                cv2.fillpoly(tmp,[pts[:j].astype(np.int32)],255, lineType=cv2.LINE_AA)
    print('total:{:.3f}'.format((time.time()-start)/10))
    print('a')