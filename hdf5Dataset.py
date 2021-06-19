import h5py
import torch
from torch.utils import data
from torchvision.io.image import decode_jpeg
import numpy as np
from pycocotools import mask as m
from pathlib import Path
import os
import cv2
import ast
import timeit
import time
from PIL import Image
class Hdf5Dataset(data.Dataset):
    """

    """
    def __init__(self, file_path):
        super().__init__()
        self.baseDir = Path(file_path)
        assert self.baseDir.is_dir()
        self.files = sorted(self.baseDir.glob('*.h5'), key = lambda x: int(x.parts[-1].replace('dataSet', '').replace('.h5', '')))
        self.totalData = len(self.files)*500
        self.data_info=[]

    def __len__(self):
        return self.totalData

    def __getitem__(self, item):
        fidx = item//500
        with h5py.File(self.files[fidx]) as f:
            dataset = f['group{0:05}'.format(fidx)]['{}'.format(item)]['image']
            data ={
                'image':  cv2.imdecode(np.array(dataset), flags=cv2.IMREAD_COLOR)
            }
            for k in ['polygon1', 'polygon2']:
                if k in data.keys():
                    data[k]=m.decode(ast.literal_eval(data[k]))#.tobytes().decode('utf-8')))
            # for k, val in dataset.attrs.items():
            #     data[k] = val
            #img = data['image']
            #img[:,:,1][data['polygon1']!=0]=255
            #cv2.imshow('img',img)500,2
            #cv2.waitKey(0)
            #print(img.shape)
        return data


    def getitem(self, i):
        return self.__getitem__(i)

class Hdf5DatasetNumPy(data.Dataset):
    """

    """

    def __init__(self, file_path):
        super().__init__()
        self.baseDir = Path(file_path)
        assert self.baseDir.is_dir()
        self.files = sorted(self.baseDir.glob('*.h5'),
                            key=lambda x: int(x.parts[-1].replace('dataSet', '').replace('.h5', '')))
        self.totalData = len(self.files) * 500
        self.data_info = []

    def __len__(self):
        return self.totalData

    def __getitem__(self, item):
        fidx = item // 500
        eidx = item%500
        with h5py.File(self.files[fidx]) as f:
            dataset = f['image']
            arr=np.array(dataset[eidx])
            data = {
                'image':arr[:,:,:3],
                'mask': arr[:,:,3]
            }

            # img = data['image']
            # img[:, :, 1][data['mask'] == 1]=255
            # img[:, :, 2][data['mask'] == 2] = 255
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # print(img.shape)
        return data

    def getitem(self, i):
        return self.__getitem__(i)

def func():
    #10 iteration total excution time for string type polygon: 122.989sec
    # 10 iter total excution time for string type polygon with worker 2: 50.9sec
    # 10 iteration total excution time for bytes type polygon store:125
    # 10 iter total excution time for bytes with worker 4: 49 sec
    device = torch.device("cuda")
    root = '/home/sabi/workspace/datablock/dataset_strings'
    dataset = Hdf5Dataset(root)
    #dataset = Hdf5DatasetNumPy(root)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2,pin_memory=True)

    for i in loader:
        pass
        #print(i['image'].shape)


        # print(data.keys())
if __name__ == "__main__":

    ti = timeit.timeit(func,number=10)
    print(ti)

