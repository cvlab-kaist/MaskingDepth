import os

import PIL.Image as pil 
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_spatial_gradient_kernel2d, normalize_kernel2d


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

# visualize color image
def save_color(tensor, path, name):
    color = np.split(tensor.clone().detach().permute(0,2,3,1).cpu().numpy(), tensor.shape[0], axis=0)
    for i, img in enumerate(color):
        img *= 255
        print(f'{path}/{name}_{i}.png')
        cv2.imwrite(f'{path}/{name}_{i}.png', cv2.cvtColor(np.squeeze(img.astype(np.uint8), axis=0), cv2.COLOR_BGR2RGB)) 

# visualize depth image
def save_depth(tensor, path, name, max_depth):
    depth = np.split(tensor.clone().detach().permute(0,2,3,1).cpu().numpy(), tensor.shape[0], axis=0)
    SCALE =  65535 / max_depth
    for i, dep in enumerate(depth):
        dep *= SCALE
        print(f'{path}/{name}_{i}.png')
        cv2.imwrite(f'{path}/{name}_{i}.png', dep.astype(np.uint16).reshape(dep.shape[1],dep.shape[2],1))

# visualize depth image (color form)
def save_magma_depth(tensor, path):
    img_np = tensor.detach().squeeze().cpu().numpy() 
    vmax = np.percentile(img_np, 95)
    normalizer = mpl.colors.Normalize(vmin=img_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(img_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save(path)

# model save
def save_model(log_path, epoch, models):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models:
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            to_save['height'] = opts.height
            to_save['width'] = opts.width
            to_save['use_stereo'] = opts.use_stereo
        torch.save(to_save, save_path)

# model & optimizer save
def save_component(log_path, model_name, epoch, model, optimizer):
    save_folder = os.path.join(log_path, model_name, "weights_{}".format(epoch+1))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #model_save
    for key, val in model.items():
        save_model_name = os.path.join(save_folder,"{}.pth".format(key))
        torch.save(val.module.state_dict(), save_model_name)

    #optimizer save
    save_optim_name = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(optimizer.state_dict(), save_optim_name)

# model mode change
def model_mode(model, mode = 0): 
    for m in model.values():
        if mode == 0: #TRAIN
            m.train()
        else:
            m.eval()
    
# evaluate error metric
def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# scale shift invariant loss function
def ssi_log_loss(source, target, alpha = 0.5, grad = False):
    loss = 0
    d = torch.log(source) - torch.log(target)
    batch_size = d.shape[0]
    diffs = d.view(batch_size, -1)
    mse = torch.mean(diffs**2, dim=-1)
    relative = torch.sum(diffs, dim=-1) ** 2 / diffs.shape[1] ** 2
    batch_loss = mse - alpha * relative
    loss += torch.mean(batch_loss)
    if grad:
        grad = spatial_gradient(d)
        grad_loss = torch.mean(grad**2)
        loss += grad_loss
    return loss

# scale shift invariant loss function(patchwise)
def patch_ssi_log_loss(source, target, box_valid, grad = True):
    box_valid = box_valid.view(-1,1)
    alpha = 0.5
    loss = 0
    d = torch.log(source) - torch.log(target)
    mask = torch.ones_like(box_valid)
    mask[torch.where(box_valid==-1)]=0
    mask = mask.reshape(-1,1,1,1).expand(-1,*d.shape[1:])  ## make mask shape like d
    
    d = d*mask
    batch_size = d.shape[0]
    diffs = d.view(batch_size, -1)
    mse = torch.mean(diffs ** 2, dim=-1)
    relative = torch.sum(diffs, dim=-1) ** 2 / diffs.shape[1] ** 2
    batch_loss = mse - alpha * relative
    loss += torch.mean(batch_loss)
    if grad:
        grad = spatial_gradient(d)
        grad_loss = torch.mean(grad**2)
        loss += grad_loss
    return loss

# scale shift invariant loss function (masking)
def ssi_log_mask_loss(source, target, valid_mask):
    ALPHA = 0.5
    diff_log = torch.log(target[valid_mask]) - torch.log(source[valid_mask])
    loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                    ALPHA * torch.pow(diff_log.mean(), 2))
    l1_loss = torch.abs(target[valid_mask] - source[valid_mask]).mean()
    
    return l1_loss

def ssi_log_loss_non_mean(source, target, grad = True):
    alpha = 0.5
    grad_loss=0
    d = torch.log(source) - torch.log(target)
    batch_size = d.shape[0]
    diffs = d.view(batch_size, -1)
    mse = torch.mean(diffs ** 2, dim=-1)
    relative = torch.sum(diffs, dim=-1) ** 2 / diffs.shape[1] ** 2
    batch_loss = mse - alpha * relative

    if grad:
        grad = spatial_gradient(d)
        grad_loss = grad**2

    return batch_loss, grad_loss

# extract spatial axis gradient
def spatial_gradient(input: torch.Tensor, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> torch.Tensor:
    """Compute the first order image derivative in both x and y using a Sobel
    operator.
    .. image:: _static/img/spatial_gradient.png
    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.
    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_edges.html>`__.
    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    # allocate kernel
    kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)

############################################################################## 
########################    For Self-Supervised Loss
############################################################################## 

def compute_reprojection_loss( pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim = SSIM()
    ssim.to(device)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

############################################################################## 
########################    geometry function set
############################################################################## 

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def normalize(x, dim=1): # L2 Norm
    """
    x.shape =(b,c,h,w)
    """        
    norm = (x**2).sum(dim, keepdim=True).pow(1./ 2)
    return x/(norm + 1e-7)

def check_valid_patch_region(image_size, box_size, embed_size, device):
    mask = torch.zeros(image_size).to(device)
    valid_checker = nn.Conv2d(1, 1, embed_size, embed_size, bias=False).to(device)
    with torch.no_grad():
        valid_checker.weight *=0 
        valid_checker.weight +=1
        
    mask[box_size[1]:box_size[3], box_size[0]:box_size[2]] = 1
    embed_mask = (valid_checker(mask.unsqueeze(0).unsqueeze(0)) > 0.0)
    ww = (embed_mask*1.0).squeeze().sum(dim = 0)
    hh = (embed_mask*1.0).squeeze().sum(dim = 1)

    l = torch.argmax(ww)
    t = torch.argmax(hh)
    w = ((ww>0.0)*1.0).sum()
    h = ((hh>0.0)*1.0).sum()

    if (h % 2) == 1:
        h += 1
        if (t + h) > embed_mask.shape[-2]:
            t -= 1

    if (w % 2) == 1:
        w += 1
        if (l + w) > embed_mask.shape[-1]:
            l -= 1
    
    embed_mask[:,:,int(t):int(t+h), int(l):int(l+w)] = True

    return embed_mask, [l,t,w,h]

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret