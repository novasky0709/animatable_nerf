import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        """
                Args:
                parent_cfg: 'configs/aninerf_s9p.yaml'

                train_dataset:
                    data_root: 'data/zju_mocap/CoreView_313'
                    human: 'CoreView_313'
                    ann_file: 'data/zju_mocap/CoreView_313/annots.npy'
                    split: 'train'
                """
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        """
                annots中有两个键,cams 和 ims
                cams中包括4个键，K,R,T,D
                这里R和T都是R_cw,T_cw。世界到相机的变换
                以Core_View_313为例，21个camera，1470frame
                len(annots['cams']['K'])->num of camera
                ['cams']
                K:21*3*3内参
                R,T：21*3，旋转用角轴表示的
                D：21*5，畸变distort
                ['ims']：1470*[‘ims’,'kpts2d']
                ['ims']：图片的相对路径，字符串
                ['kpts2d']:21*25*3(还不知道干啥的，猜测21-cam_num,3:xyz)
                """
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        """
            in default config view(CoreView313-training:[0,6,12,18])= [0, 6, 12, 18],begin_ith_frame = 0,frame_interval = 1
        """
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose or cfg.aninerf_animation:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        '''
        相当于取4个view的需要训练/测试的图像进ims
        train 313为例：取[0,6,12,18]这四个view的
        实验中[::i_intv]表示每隔多少取一张图像。由于i_intv = 1，所以是全部取了，S9P中取5，所以图像数量为1470/5

        annots['ims'][i:i + ni * i_intv][::i_intv]：取的是[0,60]前60张，[::1]不抽帧，仍然是前60张。
        ims_data：为前60张的每一张，是个21D元组['ims'][0,6,12,18]取出了需要的图像的路径
        最后使用ravel()拉平为1D数组,ravel()后了可以方便进行getitem()
        self.cam_inds:对应的元素的cam_index
        [0 6 12 18 0 6 12 18 0 6 12 18 ...]
        '''
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)#24*3,24：smpl24关节点的24个自由度，3：角轴
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))#parents:[24],是个整形
        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk
    #和aninerf_mesh_dataset的是一样的函数
    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        """
         wxyz：人体mesh的顶点坐标，在SMPL,6890个点，故dim(wxyz) = [6890,3]
         该坐标是在世界坐标系下的坐标，需要进一步进行坐标转换成smpl坐标系
         vertices为预处理得到的，对于每张图片，预先生成好他的smpl模型
        """
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        """
        params包括
        poses：paper中的theta,72个标量，24*3，每三个数一组，表示关节的旋转变量。其中第一个3d向量为root orientation目前全部(0,0,0),后23个3d向量分别表示23个关节的旋转
        Rh,Th表示SMPL坐标和世界坐标的旋转、位移变换,注意，这个不是相机的！是人的旋转和位移矩阵！！！R_human，T_human
        shapes：paper中的beta,每个标量取值[-1,1],影响人物模型的高矮胖瘦
        几项
        """
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        """
                Tip1:wxyz[6890,3],Th[1,3],wxyz-Th[6890,3].numpy中这种减法可自动转换
                Tip2:根据公式来看，I suppose:
                                            Th：twp（smpl坐标系到world坐标系的变换）
                                            Rh：Rpw（world坐标系到smpl坐标系的变换）
                通过该变换，点云转化到了smpl坐标系，这是deform可以成功的一个比较关键的变换
                Tip3:这里需要注意的是smpl是一个正交坐标系，world坐标系不规则,是一个场景相关的坐标系。
                    给出的是wxyz,相机坐标系下的点云，需要转换到p坐标系
        """
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        #A：24个变换矩阵
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)#[66,74,34,25]、[65,74,35,25],维度会变..?

        return wxyz, pxyz, A, pbw, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        if 'H' in cfg:
            H, W = cfg.H, cfg.W
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        """
        view这n个的：
        Ks:内参，
        Rs：旋转矩阵，
        Ts：位移（单位mm）
        Ds：5维变量，distort
        """
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        #T-blendWeight,
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
        tbw = tbw.astype(np.float32)

        wpts, ppts, A, pbw, Rh, Th = self.prepare_input(i)
        # ppts=pose points
        # wpts= world points
        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)
        '''
        给定img，msk，相机参数，世界坐标下的bounds，给出nerf需要的参数
        以nrays=1024为例
        rgb = [1024,3]
        ray_o = [1024,3]
        ray_d = [1024,3]
        near = [1024]
        far = [1024]
        coord = [1024,2] 像素坐标系的坐标
        mask_at_box = [1024] bool 类型，是否在mask里
        '''
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
