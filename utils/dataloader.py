# coding =utf-8
import os
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon.data.dataset import Dataset
from skimage import io, transform
from flowlib import read
from IO import read as ftread
from miscellaneous import output_process_status
import cv2

class MPIDataset(Dataset):
    def __init__(self, mpi_dirpath, wh=(640,360)):
        super(MPIDataset, self).__init__()
        self.dirpath = os.path.join(mpi_dirpath, 'training')
        self.rgb_dir = os.path.join(self.dirpath, 'clean')
        self.flow_dir = os.path.join(self.dirpath, 'flow')
        self.occlusion_dir = os.path.join(self.dirpath, 'occlusions')
        self.wh=wh
        self.length=0
        self.rgb_list = []
        self.flow_list = []
        self.occulusion_list = []
        self._load_cache()

        return

    def _load_cache(self,):
        subdirs = os.listdir(self.flow_dir)
        subdirs.sort()
        for i, sub in enumerate(subdirs):
        # for i, sub in enumerate(subdirs[0:2]):
            frame_dir = os.path.join(self.rgb_dir, sub)
            flow_dir = os.path.join(self.flow_dir, sub)
            occlusion_dir = os.path.join(self.occlusion_dir, sub)

            num_of_thissub = len(os.listdir(flow_dir))
            assert (len(os.listdir(occlusion_dir)) == num_of_thissub)
            assert (len(os.listdir(frame_dir)) == (num_of_thissub+1))

            for j in range(num_of_thissub):
                try:
                    rgb1_fn = os.path.join(frame_dir,'frame_%.4d.png'%(j+1))
                    rgb2_fn = os.path.join(frame_dir,'frame_%.4d.png'%(j+2))
                    flow_fn = os.path.join(flow_dir,'frame_%.4d.flo'%(j+1))
                    occlusion_fn = os.path.join(occlusion_dir,'frame_%.4d.png'%(j+1))
                    # rgb image
                    rgb1 = io.imread(rgb1_fn)
                    rgb2 = io.imread(rgb2_fn)
                    rgb1 = transform.resize(rgb1, (self.wh[1], self.wh[0]), order=3)
                    rgb2 = transform.resize(rgb2, (self.wh[1], self.wh[0]), order=3)
                    # mask
                    # mask = io.imread(occlusion_fn,  as_gray=True)
                    # mask = transform.resize(mask, (self.wh[1], self.wh[0])).astype(np.float32)/255
                    # print(mask.max())
                    # mask = io.imread(occlusion_fn,  as_gray=True)
                    # mask = transform.resize(mask, (self.wh[1], self.wh[0]), order=3).astype(np.float32)
                    # # mask = (1-mask/255.0)
                    mask = cv2.imread(occlusion_fn,0)
                    mask = cv2.resize(mask, (self.wh[0], self.wh[1])).astype(np.float32)/255.0
                    mask = 1 -mask
                    # flow
                    flow = read(flow_fn)
                    flow = transform.resize(flow, (self.wh[1], self.wh[0]))
                    flow[:,:,0] *= self.wh[1] / flow.shape[0]
                    flow[:,:,1] *= self.wh[0] / flow.shape[1]
                except Exception as e:
                    print('%s during add frame_.4%d'%(str(e),j+1))
                    continue

                self.rgb_list.append((rgb1.astype(np.float32),rgb2.astype(np.float32)))
                self.occulusion_list.append(mask.astype(np.float32))
                self.flow_list.append(flow.astype(np.float32))
            print('MPI ViedoSeq:%s added with %d tuples'%(sub, num_of_thissub))

    def __getitem__(self, idx):
        # I(t), I(t-1), M(t), F(t)
        return  self.rgb_list[idx][0], \
                self.rgb_list[idx][1], \
                self.occulusion_list[idx], \
                self.flow_list[idx]

    def __len__(self):
        return len(self.flow_list)

class FlyingthingsDataset(Dataset):
    def __init__(self, flyingthings_dirpath, wh=(640,360)):
        super(FlyingthingsDataset, self).__init__()
        self.dirpath = os.path.join(flyingthings_dirpath, 'train')
        self.rgb_dir = os.path.join(self.dirpath, 'image_clean','left')
        self.flow_dir = os.path.join(self.dirpath, 'flow','left','into_future')
        self.occlusion_dir = os.path.join(self.dirpath, 'flow_occlusions','left','into_future')
        self.wh=wh
        self.length=0
        self.rgb_list = []
        self.flow_list = []
        self.occulusion_list = []
        self._load_cache()

        return

    def _load_cache(self,):
        rgb_total_num = len(os.listdir(self.rgb_dir))
        print('loading flying things dataset')
        for _id in range(rgb_total_num - 1):
        # for _id in range(500):
            if _id% 100 == 99:
                output_process_status(float(_id)/float(rgb_total_num-1))

            rgb1_fn = os.path.join(self.rgb_dir,'%.7d.png'%_id)
            rgb2_fn = os.path.join(self.rgb_dir,'%.7d.png'%(_id+1))
            flow_fn = os.path.join(self.flow_dir,'%.7d.flo'%(_id))
            occlusion_fn = os.path.join(self.occlusion_dir, '%.7d.png'%_id)

            try:
                if not (os.path.exists(rgb1_fn) and \
                    os.path.exists(rgb2_fn) and \
                    os.path.exists(flow_fn) and \
                    os.path.exists(occlusion_fn)):
                    continue
                # rgb image
                rgb1 = io.imread(rgb1_fn)
                rgb2 = io.imread(rgb2_fn)
                rgb1 = transform.resize(rgb1, (self.wh[1], self.wh[0]), order=3)
                rgb2 = transform.resize(rgb2, (self.wh[1], self.wh[0]), order=3)
                # mask
                mask = cv2.imread(occlusion_fn,0)
                mask = cv2.resize(mask, (self.wh[0], self.wh[1])).astype(np.float32)/255.0
                mask = 1 -mask
                # flow
                flow = ftread(flow_fn)
                flow = transform.resize(flow, (self.wh[1], self.wh[0])).astype(np.float32)
                flow[:,:,0] *= self.wh[1] / flow.shape[0]
                flow[:,:,1] *= self.wh[0] / flow.shape[1]

            except Exception as e:
                print('%s during add frame_.7%d'%(str(e),_id))
                continue

            self.rgb_list.append((rgb1,rgb2))
            self.occulusion_list.append(mask)
            self.flow_list.append(flow)

        return

    def __getitem__(self, idx):
        # I(t), I(t-1), M(t), F(t)
        return  self.rgb_list[idx][0], \
                self.rgb_list[idx][1], \
                self.occulusion_list[idx], \
                self.flow_list[idx]

    def __len__(self):
        return len(self.flow_list)

class JointDataset(Dataset):
    def __init__(self, mpi_dirpath, flyingthings_dirpath, wh=(640,360)):
        super(JointDataset, self).__init__()
        self.mpi_ds = MPIDataset(mpi_dirpath, wh)
        self.ft_ds = FlyingthingsDataset(flyingthings_dirpath, wh)
        self.mpi_len = len(self.mpi_ds)
        self.ft_len = len(self.ft_ds)
        return
    
    def __getitem__(self, idx):
        # I(t), I(t-1), M(t), F(t)
        if idx < self.mpi_len:
            return self.mpi_ds[idx]
        else:
            return self.ft_ds[idx - self.mpi_len]

    def __len__(self):
        return  self.mpi_len + self.ft_len


def bfn4reconet(data_list):
    ICur_ls = [mx.nd.transpose(
                    mx.nd.array(x[0]),
                    axes=(2,0,1)) for x in data_list]
    ILast_ls = [mx.nd.transpose(
                    mx.nd.array(x[1]),
                    axes=(2,0,1)) for x in data_list]
    
    OccM_ls = [mx.nd.expand_dims(
                mx.nd.array(x[2]),
                axis=0) for x in data_list]
    
    Flow_ls = [mx.nd.transpose(
                    mx.nd.array(x[3]),
                    axes=(2,0,1)) for x in data_list]

    ICur_Tensor= mx.nd.stack(*ICur_ls,axis=0)
    ILast_Tensor= mx.nd.stack(*ILast_ls,axis=0)
    OccM_Tensor= mx.nd.stack(*OccM_ls,axis=0)
    Flow_Tensor= mx.nd.stack(*Flow_ls,axis=0)

    return ICur_Tensor, ILast_Tensor, OccM_Tensor, Flow_Tensor

if __name__ == '__main__':
    mpi_dir_fn = '/home/zcy6735/dataset/MPI'
    wh = (640, 360)

    db = MPIDataset(mpi_dirpath=mpi_dir_fn, wh=wh)
    print(len(db))
    a = db[10]
    print(a[2])




