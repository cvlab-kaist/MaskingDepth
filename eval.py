import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil 

import utils

def evaluate_metric(train_cfg, pred_depth, inputs):
    if train_cfg.real.dataset ==  'cityscape':
        metric = eval_cityscape(pred_depth, inputs)
    elif train_cfg.real.dataset == 'nyu':
        metric = eval_nyu(pred_depth, inputs)
    elif train_cfg.real.dataset == 'virtual_kitti':
        metric = eval_virtual_kitti(pred_depth, inputs)
    else:
        metric = eval_kitti(pred_depth, inputs)

    return metric

def eval_kitti(pred_depth, inputs):
    depth_pred = torch.clamp(F.interpolate(pred_depth, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    depth_pred[depth_pred < 1e-03] = 1e-03
    depth_pred[depth_pred > 80] = 80

    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    depth_pred = torch.clamp(depth_pred, min=1e-03, max=80)
    depth_errors = [*utils.compute_depth_errors(depth_gt, depth_pred)]

    return depth_errors

def eval_cityscape(pred_depth, inputs):
    return [0, 0, 0, 0, 0, 0, 0]

def eval_nyu(pred_depth, inputs):
    pred_depth  = pred_depth[:,:,45:471, 41:601]
    depth_gt    = inputs["depth_gt"][:,:,45:471, 41:601]

    pred_depth[pred_depth < 1e-03] = 1e-03
    pred_depth[pred_depth > 10] = 10

    pred_depth *= torch.median(depth_gt) / torch.median(pred_depth)
    pred_depth = torch.clamp(pred_depth, min=1e-03, max=10)
    depth_errors = [*utils.compute_depth_errors(depth_gt, pred_depth)]
    return depth_errors

def eval_virtual_kitti(pred_depth, inputs):
    pred_depth = utils.disp_to_depth(pred_depth, 1e-3, 80)[-1]
    depth_pred = torch.clamp(F.interpolate(pred_depth, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_gt = inputs["depth_gt"]
    
    depth_pred[depth_pred < 1e-3]  = 1e-3
    depth_pred[depth_pred > 80]    = 80

    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
    depth_errors = [*utils.compute_depth_errors(depth_gt, depth_pred)]
    return depth_errors

def get_eval_dict(errors):
    mean_errors = np.array(errors).mean(1)
    depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
    error_dict = {}
    for error_name, error_value in zip(depth_metric_names, mean_errors):
        error_dict[error_name] = error_value.item()
    return error_dict

def eval_metric(pred_depths, gt_depths, data):
    num_samples = len(pred_depths)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]
        min_depth = 0.001
        max_depth = (10.0  if data.dataset == 'nyu' else 80.0)

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        pred_depth[np.isinf(pred_depth)] = max_depth
        
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if data.dataset == 'nyu':
            eval_mask[45:471, 41:601] = 1
        else:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
    
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return abs_rel, sq_rel, rms, log_rms, d1, d2, d3

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

# visualize on wandb
def visualize(inputs, pred_depth, pred_depth_mask, pred_uncert, wandb, sample_num=4):
    b = pred_depth.shape[0]
    sample_num = b if b < sample_num else sample_num

    color  = F.interpolate(inputs['color'], inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners=False)

    for i in range(sample_num):
        wandb_eval_dict = {}
        val_depth = []

        # weak aug 
        if 'color' in inputs:
            #rgb image 
            img_we = color[i]
            img_we *= 255

            #disp image
            disp_weak_np = pred_depth[i].squeeze().cpu().numpy() 
            vmax = np.percentile(disp_weak_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_weak_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_weak_np)[:, :, :3] * 255).astype(np.uint8)
            disp_img_weak = pil.fromarray(colormapped_im)

            val_depth.append(wandb.Image(img_we, caption="weak_augment"))
            val_depth.append(wandb.Image(disp_img_weak, caption="weak_depthmap"))

            # gt_depth
            gt_depth = inputs['depth_gt'][i].squeeze().cpu().numpy() 
            vmax = np.percentile(gt_depth, 95)
            normalizer = mpl.colors.Normalize(vmin=gt_depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(gt_depth)[:, :, :3] * 255).astype(np.uint8)
            gt_depth = pil.fromarray(colormapped_im)

            val_depth.append(wandb.Image(gt_depth, caption="gt_depth"))

        #strong aug 
        if 'color_aug' in inputs:
            #rgb image 
            img_st = inputs['color_aug'][i].clone().detach().permute(1,2,0).cpu().numpy() 
            img_st *= 255     

            #disp image
            disp_strong_np = pred_depth_mask[i].squeeze().cpu().numpy() 
            vmax = np.percentile(disp_strong_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_strong_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_strong_np)[:, :, :3] * 255).astype(np.uint8)
            disp_img_strong = pil.fromarray(colormapped_im)
        
            val_depth.append(wandb.Image(img_st, caption="strong_augment"))
            val_depth.append(wandb.Image(disp_img_strong, caption="strong_depthmap"))
        
        wandb_eval_dict["validation depthmap"] = val_depth

        # # uncertainty
        if not(pred_uncert == None):
            uncert_np = pred_uncert[i].squeeze().cpu().numpy()
            umax = uncert_np.max()
            umin = uncert_np.min()
            unormalizer = mpl.colors.Normalize(vmin=umin, vmax=umax)
            umapper = cm.ScalarMappable(norm=unormalizer, cmap='hot')
            ucolormapped_im = (umapper.to_rgba(uncert_np)[:, :, :3] * 255).astype(np.uint8)
            uncert_img = pil.fromarray(ucolormapped_im)
            wandb_eval_dict["uncertainty map"] = [wandb.Image(uncert_img, caption="Uncertainty map")]

        # confidence
        
        wandb.log(wandb_eval_dict)
