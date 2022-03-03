import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.bw_latent = nn.Embedding(cfg.num_train_frame + 1, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)#bw对于某一个点，就是24个标量

        if cfg.aninerf_animation:
            self.novel_pose_bw = BackwardBlendWeight()

            if 'init_aninerf' in cfg:
                net_utils.load_network(self,
                                       'data/trained_model/deform/' +
                                       cfg.init_aninerf,
                                       strict=False)
    '''
    pose_pts: n_batch, n_point, 3
    ind : n_batch,value(I guess)
    输出：bw_net的input（191D的向量）s
    '''
    def get_bw_feature(self, pts, ind):
        pts = embedder.xyz_embedder(pts)#3->63D,PE操作
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)#输入index 固定输出128D index
        # 这行，行将一个[1,128]，和pts（[1,63,15446]）的三通道展成一个维度，如[1,128,15446]
        #latent[..., None]:[1,128,1]
        #latent.shape is [1,128]; *latent.shape: 1,128,相当于把一个数组，解开挨个作为参数传入
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)#得到的是一个191D的bw-feature；由点坐标（3d->63D）+隐变量（128D）,cat而成
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)#fc后是一个24D的向量(原文），现在改成了一个3D的偏移(e.x.[1,3,7457])
        '''
        Equation(5),see paper
        '''
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw

    def pose_points_to_tpose_points(self, pose_pts, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        #在pose坐标下做的前面的with no grand 部分是做了一个筛选，现在开始采样pbw
        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        init_pbw = init_pbw[:, :24]

        # neural blend weights of points at i
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw,
                                     batch['bw_latent_index'])
        else:
            # go this way at first stage(without animation)
            pbw = self.calculate_neural_blend_weights(
                pose_pts, init_pbw, batch['latent_index'] + 1)
        # 这里得到的pbw是derta pbw，3d向量(e.x.[1,3,7457])
        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])
        #t坐标下的所有采样点和他们对应的p坐标下的derta bw
        return tpose, pbw

    def calculate_alpha(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        pnorm = init_pbw[:, 24]
        norm_th = 0.1
        pind = pnorm < norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pose_pts = pose_pts[pind][None]

        # transform points from the pose space to the tpose space
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = self.tpose_human.calculate_alpha(tpose)
        alpha = alpha[0, 0]

        n_batch, n_point = wpts.shape[:2]
        full_alpha = torch.zeros([n_point]).to(wpts)
        full_alpha[pind[0]] = alpha

        return full_alpha

    def forward(self, wpts, viewdir, dists, batch):
        #wpts:[65536,3],viewdire[65536,3],dists[65536]
        #batch:dict_keys(['rgb', 'occupancy', 'ray_o', 'ray_d', 'near', 'far', 'mask_at_box', 'A', 'pbw', 'tbw', 'pbounds', 'wbounds', 'tbounds', 'R', 'Th', 'H', 'W', 'latent_index', 'bw_latent_index', 'frame_index', 'cam_ind']),tpose出来的
        # transform points from the world space to the pose space
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        with torch.no_grad():

            #batch['pbw'] [1,68,74,26,25]可变的，在pose坐标下的blend weight；cv估计出的
            # init_pbw:[1,25,65536]，得到了这65536个点的bw
            init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                batch['pbounds'])
            pnorm = init_pbw[:, -1]#init_pbw的最后一列元素为pnorm
            norm_th = cfg.norm_th#0.05
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True#p index,[1,65536]
            #进行了一波筛选...
            pose_pts = pose_pts[pind][None]#某一次:[1,7503,3]
            viewdir = viewdir[pind[0]]
            dists = dists[pind[0]]
        # transform points from the pose space to the tpose space
        # t坐标下的所有采样点和他们对应的p坐标下的bw，现在这个pbw没意义了
        # transform points from the pose space to the tpose space
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)
        # calculate neural blend weights of points at the tpose space，在t坐标下，算一下他们的bw，由tbw采样得到的
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)#这里index==0

        viewdir = viewdir[None]
        ind = batch['latent_index']
        alpha, rgb = self.tpose_human.calculate_alpha_rgb(tpose, viewdir, ind)

        inside = tpose > batch['tbounds'][:, :1]
        inside = inside * (tpose < batch['tbounds'][:, 1:])
        outside = torch.sum(inside, dim=2) != 3
        alpha = alpha[:, 0]
        alpha[outside] = 0

        alpha_ind = alpha.detach() > cfg.train_th
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind][None]
        tbw = tbw.transpose(1, 2)[alpha_ind][None]

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * dists)

        rgb = torch.sigmoid(rgb[0])
        alpha = raw2alpha(alpha[0], dists)

        raw = torch.cat((rgb, alpha[None]), dim=0)
        raw = raw.transpose(0, 1)

        n_batch, n_point = wpts.shape[:2]
        raw_full = torch.zeros([n_batch, n_point, 4], dtype=wpts.dtype, device=wpts.device)
        raw_full[pind] = raw

        ret = {'pbw': pbw, 'tbw': tbw, 'raw': raw_full}

        return ret


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()
        #以train_frame为编号创建latent code
        self.nf_latent = nn.Embedding(cfg.num_train_frame, 128)#隐变量per-frame latent code

        self.actvn = nn.ReLU()

        input_ch = 63
        D = 8
        W = 256
        self.skips = [4]
        self.pts_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.alpha_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        self.latent_fc = nn.Conv1d(384, W, 1)#256+128,256为feature_fc,128为latent code ,encode the state of human appearance frame i
        self.view_fc = nn.Conv1d(283, W // 2, 1)#256+27,256为latent_fc输出，27为Positional Encoding后的ray_direction
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)#最终输出颜色

    def calculate_alpha(self, nf_pts):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)
        return alpha

    def calculate_alpha_rgb(self, nf_pts, viewdir, ind):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.nf_latent(ind)
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        return alpha, rgb


class BackwardBlendWeight(nn.Module):
    def __init__(self):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_eval_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw
