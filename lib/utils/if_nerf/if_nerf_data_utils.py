import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh


def get_rays_within_bounds_test(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    '''
    K,R,T:相机的内外参数，这里R和T都是R_cw,T_cw。世界到相机的变换
    因为view = 4；所以rays_o求得的是四个相机的光心在世界坐标系下的坐标
    1.世界坐标到相机坐标的转换
    P_c = R_cw*P_w + T_cw
    P_w = R_cw.inverse()*(P_c-T_cw)
    取P_c = 0
    rays_o = -R_cw.inverse()*T_cw
    '''
    rays_o = -np.dot(R.T, T).ravel()#ravel():相当于flatten()的操作
    # calculate the world coodinates of pixels
    #这几行代码值得学习，meshgrid，后面接1等操作，得到方向，broadcast_to等操作

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)#得到逐像素的世界坐标
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]#得到方向向量
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)#得到单位方向向量
    rays_o = np.broadcast_to(rays_o, rays_d.shape)#值得学习的操作。
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


# def get_near_far(bounds, ray_o, ray_d):
#     """calculate intersections with 3d bounding box"""
#     norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
#     viewdir = ray_d / norm_d
#     viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
#     viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
#     tmin = (bounds[:1] - ray_o[:1]) / viewdir
#     tmax = (bounds[1:2] - ray_o[:1]) / viewdir
#     t1 = np.minimum(tmin, tmax)
#     t2 = np.maximum(tmin, tmax)
#     near = np.max(t1, axis=-1)
#     far = np.min(t2, axis=-1)
#     mask_at_box = near < far
#     near = near[mask_at_box] / norm_d[mask_at_box, 0]
#     far = far[mask_at_box] / norm_d[mask_at_box, 0]
#     return near, far, mask_at_box


def get_near_far(bounds, ray_o, ray_d):
    '''
    注：补none的操作也是值得学习的，这是python原生的squeeze
    np.array([-0.01, 0.01]):[2,]
    np.array([-0.01, 0.01])[:,None]:[2,1]
    bounds:[2,3]
    bounds[None]:[1,2,3]
    ray_o:[1024,3]
    ray_o[:, None]:[1024,1,3]
    '''
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]#[2,3]+[2,1] = [2,3],[2,1]在最低维度上被复制为[2,3]并相加

    nominator = bounds[None] - ray_o[:, None]#[1,2,3]+[1024,1,3] = [1024,2,3]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)#[1024,2,3]/[1024,1,3] = [1024,2,3]
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

"""
input:
img：图像（RGB)
msk：2值msk
K：相机内参
R，T：相机外参
bounds：smpl人物模型的点云在世界坐标系下的包围盒
nrays：发射光线的数量（？）
split：train or test
"""


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if cfg.mask_bkgd:
        img[bound_mask != 1] = 0

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    '''
    如果是训练集，随机进行采样，如果是测试集，全部都要算rgb
    '''
    if split == 'train':
        nsampled_rays = 0 #已采集的光线
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)#身上采集的ray数量
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)#脸上采集的ray数量
            n_rand = (nrays - nsampled_rays) - n_body - n_face#剩余的随机采集的ray数量

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)
                '''
                coord 得到所有要采集的坐标的像素值（u，v）坐标
                coord[1024,2]
                ray_o[512,512,3]
                ray_o_[1024,3]

                A = np.arange(16).reshape(4,4)
                print(A)
                msk1 = [0,1,2,3]
                msk2 = [3,2,1,0]
                A = A[msk1,msk2]
                print(A)
                这个代码值得学习，做一个挑选需要shoot ray的像素的操作
                '''

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            # get_near_far 这个没细看，返回的是3个[1024]数组，记录每个ray的far，near,mask_at_box返回bool
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        # get_near_far 这个没细看，返回的是3个[1024]数组，记录每个ray的far，near,mask_at_box返回bool
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)
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
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_rays_within_bounds(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_acc(coord, msk):
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    joints和parents是全局唯一的，poses是每张照片一个的
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    返回24个变换矩阵
    """
    # 将旋转向量全部转化为旋转矩阵
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    return transforms


def padding_bbox(bbox, img):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size=80, max_size=88):
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    min_size = int(min(min_size, 0.8 * min_HW))
    H_size = np.random.randint(min_size, max_size)
    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coord = np.argwhere(msk == 1)
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bounds(xyz):
    '''

    Args:
        xyz: [4890,3]人体点云数据
    Returns:
        [2,3],xyz上的bounds

    '''
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= cfg.box_padding
    max_xyz += cfg.box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds


def prepare_sp_input(xyz):
    # obtain the bounds for coord construction
    bounds = get_bounds(xyz)
    # construct the coordinate
    dhw = xyz[:, [2, 1, 0]]
    min_dhw = bounds[0, [2, 1, 0]]
    max_dhw = bounds[1, [2, 1, 0]]
    voxel_size = np.array(cfg.voxel_size)
    coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
    # construct the output shape
    out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh, bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img
