import os
import numpy as np
import h5py as hf
import pdb
import pandas as pd
import cv2
import time
import ast
from pycocotools import mask as m
from torchvision.io.image import read_file, decode_jpeg


def make_dataset_attr():
    root = '/home/sabi/workspace/ganhackerton/dataset'
    with os.scandir(root) as entries:
        fpathes = [os.path.join(root, fname) for fname in entries if os.path.isfile(fname)]
    dfall = pd.concat([pd.read_csv(fpath) for fpath in fpathes[:2]]).drop('Unnamed: 0',axis=1)
    basename = '/main/dataset'
    pts = np.zeros((1000,2))
    tmp = np.zeros((512,512),dtype=np.uint8)
    keys = [ k for k in dfall.columns if 'exif' not in k and 'polygon' not in k]
    utf8_type = hf.string_dtype('utf-8')
    start = time.time()
    with open('datasetInfo.txt', 'w') as dataSetInfo:
        dataSetInfo.write('*.h5 file Structure:\n')
        dataSetInfo.write('Each group has 500 dataset, each of the dataset has an image in bytes format and its labels in attributes.\n')
        dataSetInfo.write('An image in dataset is stored as bytes so it is needed to be decode.\n')
        dataSetInfo.write('image size: 512x512x3\n')
        #dfall.shape[0]//500
        for j in range(dfall.shape[0]//500):
            fname = 'dataSet{0:03d}.h5'.format(j)
            dsname = os.path.join('/home/sabi/workspace/datablock/dataset', fname)
            dataSetInfo.write(fname+'\n')
            hfg = hf.File(dsname, 'w')
            grp = hfg.create_group('group{0:03d}'.format(j))
            for i in range(500*j, 500*(j+1)):
                item = dfall.iloc[i]
                with open(item['path'].replace(basename, root), 'rb') as f:
                    bimg = f.read()
                    dset = grp.create_dataset('{}'.format(i), data=np.frombuffer(bimg, dtype=np.uint8))
                    try:
                        for n in [1,2]:
                            name = 'polygon{}'.format(n)
                            tmp.fill(0)
                            poly = ast.literal_eval(item[name])
                            pl_list =[]
                            for pl in poly:
                                j = 0
                                for p in pl:
                                    x, y = p['x'], p['y']
                                    pts[j] = x, y
                                    j += 1
                                pl_list.append(pts[:j].astype(np.int32))
                            cv2.fillPoly(tmp, pl_list, 255, lineType=cv2.LINE_AA)
                            mask = np.array(tmp > 0, dtype=bool, order='F')
                            rle_mask = m.encode(mask)
                            dset.attrs[name] = str(rle_mask).encode('utf-8')
                    except KeyError as e:
                        pass

                    for k in keys:
                        data = item[k]
                        dset.attrs[k] = np.array(data, dtype = utf8_type if isinstance(data, str) else float)

            hfg.close()

def make_numpy_data_set():
    root = '/home/sabi/workspace/ganhackerton/dataset'
    with os.scandir(root) as entries:
        fpathes = [os.path.join(root, fname) for fname in entries if os.path.isfile(fname)]
    dfall = pd.concat([pd.read_csv(fpath) for fpath in fpathes[:2]]).drop('Unnamed: 0', axis=1)
    basename = '/main/dataset'
    pts = np.zeros((1000, 2))
    tmp = np.zeros((512, 512), dtype=np.uint8)
    keys = [k for k in dfall.columns if 'exif' not in k and 'polygon' not in k]
    utf8_type = hf.string_dtype('utf-8')
    start = time.time()

    for j in range(dfall.shape[0]//500):
        dsname = '/home/sabi/workspace/datablock/dataset/dataSet{}.h5'.format(j)
        hfg = hf.File(dsname, 'w')
        dataset = hfg.create_dataset('image',(500,512,512,4),dtype=np.uint8)
        for i in range(500):
            item = dfall.iloc[i+500*j]
            img = cv2.imread(item['path'].replace(basename, root))
            try:
                tmp.fill(0)
                for n in [1, 2]:
                    name = 'polygon{}'.format(n)
                    poly = ast.literal_eval(item[name])
                    pl_list = []
                    for pl in poly:
                        len = 0
                        for p in pl:
                            x, y = p['x'], p['y']
                            pts[len] = x, y
                            len += 1
                        pl_list.append(pts[:len].astype(np.int32))
                    cv2.fillPoly(tmp, pl_list, n, lineType=cv2.LINE_AA)

                # img[:,:,1][tmp==1]=255
                # img[:, :, 2][tmp == 2] = 255
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                dataset[i,:,:,:3]=img
                dataset[i,:,:,3]=tmp

                '''
                mask = np.array(tmp > 0, dtype=bool, order='F')
                rle_mask = m.encode(mask)
                '''

            except KeyError as e:
                pass
            '''
            for k in keys:
                data = item[k]
                dset.attrs[k] = np.array(data, dtype=utf8_type if isinstance(data, str) else float)
            '''

        hfg.close()

if __name__=='__main__':
    start = time.time()
    if 1:
        make_dataset_attr()
    else:
        make_numpy_data_set()



    print('total:{:.3f}'.format((time.time()-start)/10))
    print('a')
