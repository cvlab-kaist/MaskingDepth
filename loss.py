import torch
import torch.nn.functional as F

import utils

TRAIN   = 0
EVAL    = 1

# total loss(sup)
def compute_loss(inputs, model, train_cfg, mode = TRAIN):
    losses = {}
    total_loss = 0

    pred_depth, full_features, fusion_features = model_forward(inputs['color'], model)
    
    ### compute supervised loss 
    if train_cfg.data.dataset in ['nyu'] or train_cfg.unlabeled_data.dataset in ['nyu']:
        losses['sup_loss'] = compute_sup_loss(pred_depth, inputs['depth_gt'])

    else:
        pred_depth = F.interpolate(pred_depth, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
        losses['sup_loss'] = compute_sup_loss(pred_depth, inputs['depth_gt'], (inputs['depth_gt'] > 0).detach())
    
    ### make uncertainty map
    if 'uncert_decoder' in model.keys():
        pred_uncert = uncert_forward(fusion_features, model)
        if train_cfg.data.dataset == 'nyu' or train_cfg.unlabeled_data.dataset == 'nyu':
            losses['uncert_loss'] = compute_uncert_loss(pred_uncert, pred_depth, inputs['depth_gt'])
        else:
            pred_uncert = F.interpolate(pred_uncert, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
            losses['uncert_loss'] = compute_uncert_loss(pred_uncert, pred_depth, inputs['depth_gt'], (inputs['depth_gt'] > 0).detach())
    else: 
        pred_uncert = None 

    ### compute consistency loss
    if not(train_cfg.d_consistency == 0) or not(train_cfg.f_consistency == 0):
        
        #### make K-way augmented depth map
        pred_depth_mask, mask_features, _, = model_forward(inputs['color_aug'], model, crop, K = train_cfg.K)

        if not(train_cfg.data.dataset == 'nyu' or train_cfg.unlabeled_data.dataset == 'nyu'):
            pred_depth_mask = F.interpolate(pred_depth_mask, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)

        ### consistency loss between weak depth map and strong depth map
        if not(train_cfg.d_consistency == 0):
            losses['consistency_loss'] = train_cfg.d_consistency * compute_adaptive_consistency_loss(pred_depth, pred_depth_mask, pred_uncert)
        if not(train_cfg.f_consistency == 0):
            losses['feature_consistency_loss'] = train_cfg.f_consistency * compute_feature_consistency_loss(full_features, mask_features, model)
        
    else:
        if mode == EVAL:
            pred_depth_mask, _, _ = model_forward(inputs['color_aug'], model, crop, K = train_cfg.K)
        else:
            pred_depth_mask = None


    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, pred_depth, pred_uncert, pred_depth_mask

# total loss(semi)
def compute_semi_loss(label, unlabel, model, train_cfg, mode = TRAIN):
    losses = {}
    total_loss = 0
    
    
    label_pred_depth, label_full_features, label_fusion_features = model_forward(label['color'], model)


    if train_cfg.dataset == 'nyu':
        losses['sup_loss'] = compute_sup_loss(label_pred_depth, label['depth_gt'])
    else:
        label_pred_depth = F.interpolate(label_pred_depth, label['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
        losses['sup_loss'] = compute_sup_loss(label_pred_depth, label['depth_gt'], (label['depth_gt'] > 0).detach())
        
    if 'uncert_decoder' in model.keys():
        label_pred_uncert = uncert_forward(label_fusion_features, model)
        if train_cfg.dataset == 'nyu':
            losses['uncert_loss'] = compute_uncert_loss(label_pred_uncert, label_pred_depth, label['depth_gt'])
        else:
            label_pred_uncert = F.interpolate(label_pred_uncert, label['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
            losses['uncert_loss'] = compute_uncert_loss(label_pred_uncert, label_pred_depth, label['depth_gt'], (label['depth_gt'] > 0).detach())

    if not(train_cfg.d_consistency == 0) or not(train_cfg.f_consistency == 0) or (train_cfg.self_sup==True):
        unlabel_pred_depth, unlabel_full_features, unlabel_fusion_features = model_forward(unlabel['color'], model)
        unlabel_pred_uncert = uncert_forward(unlabel_fusion_features, model) if 'uncert_decoder' in model.keys() else None

        unlabel_pred_depth_mask, unlabel_mask_features, _ = model_forward(unlabel['color_aug'], model, train_cfg.K)

        if not(train_cfg.d_consistency == 0):
            losses['consistency_loss'] = train_cfg.d_consistency * compute_adaptive_consistency_loss(unlabel_pred_depth_mask, unlabel_pred_depth, unlabel_pred_uncert)

        if not(train_cfg.f_consistency == 0):
            losses['feature_consistency_loss'] = train_cfg.f_consistency * compute_feature_consistency_loss(unlabel_full_features, unlabel_mask_features, model)
        
        if train_cfg.self_sup == True:
            losses['selfsup_loss'] = compute_selfsup_loss(unlabel_pred_depth, unlabel, train_cfg)
    
    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, unlabel_pred_depth, unlabel_pred_uncert

############################################################################## 
########################    model forward
############################################################################## 

# main network forward
def model_forward(inputs, model, K=0.0):  
    pred_depth, features, fusion_features = model['depth'](inputs, K)
    return pred_depth, features, fusion_features

# (additional) uncertainty network forward
def uncert_forward(fusion_features, model):
    pred_uncert = model['uncert_decoder'](fusion_features)
    return pred_uncert

############################################################################## 
########################    loss function set
############################################################################## 
  
def compute_sup_loss(pred_depth, gt_depth, mask=None): 
    if mask == None:
        loss = torch.abs(pred_depth - gt_depth.detach()).mean()
    else:
        loss = torch.abs(pred_depth[mask] - gt_depth.detach()[mask]).mean()
    return loss

def compute_sup_mask_loss(pred_depth, gt_depth): 
    pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:], mode="bilinear", align_corners = False)
    return utils.ssi_log_mask_loss(pred_depth+1, gt_depth.detach()+1, (gt_depth > 0).detach())

def compute_uncert_loss(pred_uncert, pred_depth, gt_depth, mask=None):
    if mask == None:
        loss = (torch.abs(pred_depth.detach() - gt_depth.detach()) * torch.exp(-pred_uncert) + 0.1*pred_uncert).mean()
    else:
        loss = (torch.abs(pred_depth[mask].detach() - gt_depth[mask].detach()) * torch.exp(-pred_uncert[mask]) + 0.3*pred_uncert[mask]).mean()
    return loss

def compute_consistency_loss(pred_depth_mask, pred_depth_full):
    return torch.abs(pred_depth_mask - pred_depth_full.detach()).mean()

def compute_adaptive_consistency_loss(pred_depth_mask, pred_depth_full, pred_uncert):
    if pred_uncert == None:
        return torch.abs(pred_depth_mask - pred_depth_full.detach()).mean()

    else:
        return (torch.abs(pred_depth_mask - pred_depth_full.detach()) * torch.exp(-pred_uncert.detach())).mean()

def compute_feature_consistency_loss(full_features, masking_features, model):
    loss = 0
    criterion = torch.nn.MSELoss()

    for i, (Key, Query) in enumerate(zip(full_features, masking_features)):
        Query = Query.transpose(1,2).flatten(0,1)
        Key   = Key.transpose(1,2).flatten(0,1).detach()
        
        if 'mlp_head' in model.keys():
            Query = model['mlp_head'](Query, i)        
            
        feat_Q = utils.normalize(Query, dim=1) 
        feat_K = utils.normalize(Key,   dim=1)
        
        loss += criterion(feat_Q, feat_K)
        
    loss /= len(full_features)
    return loss
