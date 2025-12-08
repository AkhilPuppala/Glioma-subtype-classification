# Main_DTFD_MIL.py
import os
import time
import json
import re
import torch
import numpy as np
import random
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# --- Import your model components and utils (adjust paths as in your repo) ---
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from Model.network import Classifier_1fc, DimReduction
from utils import get_cam_1d, eval_metric, eval_metric_
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='abc')
parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--label_col', default='label', type=str)
parser.add_argument('--isIHC', default=False, type=bool)
parser.add_argument('--k_start', default=-1, type=int)
parser.add_argument('--k_end', default=-1, type=int)
parser.add_argument('--isPar', default=True, type=bool)
parser.add_argument('--splits_dir', default='', type=str)
parser.add_argument('--dataset_csv', default='', type=str)
parser.add_argument('--num_cls', default=3, type=int)
parser.add_argument('--data_dir', default='./features', type=str)
parser.add_argument('--in_chn', default=384, type=int)
parser.add_argument('--mDim', default=384, type=int)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[90]', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--log_dir', default='./results', type=str)
parser.add_argument('--train_show_freq', default=200, type=int)
parser.add_argument('--droprate', default=0.0, type=float)
parser.add_argument('--droprate_2', default=0.0, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-3, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_true')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=1, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------
# Utility classes / funcs
# -------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=85):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_metric = float('-inf')
        self.early_stop = False
    def __call__(self, val_metric, epoch):
        if epoch < self.stop_epoch: return False
        if val_metric > self.best_metric:
            self.best_metric = val_metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def print_log(tstr, f):
    try:
        f.write('\n'); f.write(tstr); f.flush()
    except Exception:
        pass
    print(tstr)

def transform_state_dict(state_dict, util_dict):
    # compatibility helper for loading older models (keeps same as before)
    for j in util_dict.keys():
        var = state_dict[j]
        new_state_dict = {}
        for k, v in var.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        # remove 3-channel head if present (legacy)
        keys_list = list(new_state_dict.keys())
        for i in keys_list:
            try:
                if new_state_dict[i].shape[0] == 3:
                    new_state_dict[i+'_not_to_use'] = new_state_dict[i]
                    del new_state_dict[i]
            except Exception:
                pass
        state_dict[j] = new_state_dict
    return state_dict

# -------------------------
# Data reorganization helpers
# -------------------------
def reOrganize_mDATA(dataset_csv, fold_csv, set_type, label_name='label'):
    mDATA_slides = pd.read_csv(fold_csv)
    mDATA_label = pd.read_csv(dataset_csv)

    if set_type not in mDATA_slides.columns:
        raise KeyError(f"Expected column '{set_type}' in splits CSV ({fold_csv}). Available columns: {list(mDATA_slides.columns)}")

    temp_SlideNames = mDATA_slides[set_type].dropna().tolist()

    possible_slide_cols = ['slide_id', 'SlideID', 'Slide_Id', 'slide', 'Slide', 'filename', 'file', 'image']
    slide_col = None
    for col in possible_slide_cols:
        if col in mDATA_label.columns:
            slide_col = col; break
    if slide_col is None:
        raise KeyError(f"Could not find a slide id column in dataset CSV ({dataset_csv}). Tried: {possible_slide_cols}. Available columns: {list(mDATA_label.columns)}")

    mDATA = mDATA_label[mDATA_label[slide_col].isin(temp_SlideNames)]

    if label_name not in mDATA.columns:
        raise KeyError(f"Label column '{label_name}' not found in dataset CSV ({dataset_csv}). Available columns: {list(mDATA.columns)}")

    if mDATA[label_name].dtype == object:
        unique_labels = sorted(mDATA[label_name].unique())
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        mDATA.loc[:, label_name] = mDATA[label_name].map(label_to_int)
        print(f"Auto-mapped label values to integers (fixed): {label_to_int}")

    SlideNames = mDATA[slide_col].tolist()
    Label = mDATA[label_name].tolist()

    if len(SlideNames) == 0:
        temp_basenames = [os.path.splitext(s)[0] for s in temp_SlideNames]
        mDATA['_basename_'] = mDATA[slide_col].apply(lambda x: os.path.splitext(str(x))[0])
        mDATA_basename = mDATA[mDATA['_basename_'].isin(temp_basenames)]
        if len(mDATA_basename) > 0:
            SlideNames = mDATA_basename[slide_col].tolist()
            Label = mDATA_basename[label_name].tolist()

    return SlideNames, Label

# -------------------------
# TEST function (as in your file) - signature preserved
# returns: auc_1, mF1_1, (gPred_0, gPred_1), (gt_0, gt_1)
# -------------------------
def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch,
                                             criterion=None, params=None, f_log=None, writer=None, numGroup=3,
                                             total_instance=3, distill='MaxMinS'):

    # set eval mode for components
    try:
        classifier.eval(); attention.eval(); dimReduction.eval(); UClassifier.eval()
    except Exception:
        pass

    sl_names = []
    SlideNames, Label = mDATA_list
    instance_per_group = max(1, total_instance // max(1, numGroup))

    test_loss0 = AverageMeter(); test_loss1 = AverageMeter()
    gPred_0 = torch.FloatTensor().to(params.device); gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device); gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():
        numSlides = len(SlideNames)
        numIter = max(1, (numSlides + params.batch_size_v - 1) // params.batch_size_v)
        tIDX = list(range(numSlides))

        for idx in range(numIter):
            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            if len(tidx_slide) == 0: continue

            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)

            batch_feat = []
            for sst in tidx_slide:
                pt_path = os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(SlideNames[sst]))
                try:
                    loaded = torch.load(pt_path)
                except Exception as e:
                    print(f'Error loading {pt_path}: {e}')
                    loaded = torch.zeros((1, params.in_chn))
                loaded = loaded.to(params.device)
                batch_feat.append(loaded)

            sl_names.extend([SlideNames[sst] for sst in tidx_slide])

            for tidx, tfeat in enumerate(batch_feat):
                if tfeat is None or tfeat.numel() == 0: continue
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)
                AA = attention(midFeat, isNorm=False).squeeze(0)

                allSlide_pred_softmax = []
                allSlide_pred_logits = []

                for jj in range(max(1, params.num_MeanInference)):
                    feat_index = list(range(tfeat.shape[0] if hasattr(tfeat, 'shape') else 1))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list if len(sst) > 0]
                    if len(index_chunk_list) == 0:
                        index_chunk_list = [feat_index]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        if len(tindex) == 0:
                            continue
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        idx_tensor = idx_tensor.clamp(min=0, max=max(0, midFeat.shape[0]-1))
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)
                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        if tmidFeat.shape[0] == 0:
                            continue
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
                        tPredict = classifier(tattFeat_tensor)
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

                        cls_idx = -1 if patch_pred_softmax.shape[1] > 1 else 0
                        _, sort_idx = torch.sort(patch_pred_softmax[:, cls_idx], descending=True)

                        k = max(1, instance_per_group)
                        topk_idx_max = sort_idx[:k].long() if sort_idx.numel() > 0 else torch.LongTensor([]).to(params.device)
                        topk_idx_min = sort_idx[-k:].long() if sort_idx.numel() > 0 else torch.LongTensor([]).to(params.device)

                        if distill == 'MaxMinS':
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0) if topk_idx_min.numel()>0 else topk_idx_max
                            if topk_idx.numel() == 0:
                                d_inst_feat = tattFeat_tensor
                            else:
                                topk_idx = topk_idx.clamp(min=0, max=max(0, tmidFeat.shape[0]-1))
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx.to(tmidFeat.device))
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            if topk_idx_max.numel() == 0:
                                d_inst_feat = tattFeat_tensor
                            else:
                                topk_idx_max = topk_idx_max.clamp(min=0, max=max(0, tmidFeat.shape[0]-1))
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max.to(tmidFeat.device))
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    if len(slide_d_feat) == 0:
                        tAA_full = torch.softmax(AA, dim=0)
                        tattFeats_full = torch.einsum('ns,n->ns', midFeat, tAA_full)
                        tattFeat_tensor_full = torch.sum(tattFeats_full, dim=0).unsqueeze(0)
                        slide_d_feat = [tattFeat_tensor_full]

                    try:
                        slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    except Exception:
                        slide_d_feat = slide_d_feat[0]

                    if len(slide_sub_preds) == 0:
                        agg_att = torch.sum(torch.einsum('ns,n->ns', midFeat, torch.softmax(AA, dim=0)), dim=0).unsqueeze(0)
                        tPredict = classifier(agg_att)
                        slide_sub_preds = [tPredict]
                        slide_sub_labels = [tslideLabel]

                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_logits.append(gSlidePred)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                if len(allSlide_pred_logits) == 0:
                    if len(allSlide_pred_softmax) > 0:
                        allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                        allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                        mean_logits = torch.log(allSlide_pred_softmax + 1e-12)
                    else:
                        mean_logits = torch.zeros((1, params.num_cls)).to(params.device)
                else:
                    mean_logits = torch.mean(torch.stack(allSlide_pred_logits, dim=0), dim=0)
                    if mean_logits.dim() == 1:
                        mean_logits = mean_logits.unsqueeze(0)

                try:
                    loss1 = F.cross_entropy(mean_logits, tslideLabel)
                except Exception:
                    mean_soft = torch.softmax(mean_logits, dim=1)
                    loss1 = F.nll_loss(torch.log(mean_soft + 1e-12), tslideLabel)

                test_loss1.update(loss1.item(), 1)

                probs = torch.softmax(mean_logits, dim=1)
                gPred_1 = torch.cat([gPred_1, probs], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

    if gPred_0.numel() > 0:
        try: gPred_0 = torch.softmax(gPred_0, dim=1)
        except Exception: pass

    if params.num_cls == 2:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)
    else:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric_(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric_(gPred_1, gt_1)

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    try:
        writer.add_scalar('auc_0 ', auc_0, epoch); writer.add_scalar('auc_1 ', auc_1, epoch)
        writer.add_scalar('F1_0 ', mF1_0, epoch); writer.add_scalar('F1_1 ', mF1_1, epoch)
        writer.add_scalar('Acc_0 ', macc_0, epoch); writer.add_scalar('Acc_1 ', macc_1, epoch)
    except Exception:
        pass

    return auc_1, mF1_1, (gPred_0, gPred_1), (gt_0, gt_1)

# -------------------------
# TRAIN function (as in your file) - signature preserved
# -------------------------
def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier, optimizer0, optimizer1,
                                    epoch, ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=15,
                                    distill='MaxMinS'):

    SlideNames_list, Label_dict = mDATA_list

    try:
        classifier.train(); dimReduction.train(); attention.train(); UClassifier.train()
    except Exception:
        pass

    instance_per_group = max(1, total_instance // max(1, numGroup))
    Train_Loss0 = AverageMeter(); Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = max(1, (numSlides + params.batch_size - 1) // params.batch_size)
    tIDX = list(range(numSlides)); random.shuffle(tIDX)

    for idx in range(numIter):
        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]
        if len(tidx_slide) == 0: continue

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []; slide_sub_preds = []; slide_sub_labels = []

            tfeat_path = os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(tslide))
            try:
                tfeat_tensor = torch.load(tfeat_path)
            except Exception as e:
                print(f'Warning: failed to load {tfeat_path}: {e}')
                tfeat_tensor = torch.zeros((1, params.in_chn))
            if not isinstance(tfeat_tensor, torch.Tensor):
                tfeat_tensor = torch.tensor(tfeat_tensor, dtype=torch.float32)
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_count = max(1, tfeat_tensor.shape[0])
            feat_index = list(range(feat_count)); random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list if len(sst) > 0]
            if len(index_chunk_list) == 0: index_chunk_list = [feat_index]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                if len(tindex) == 0:
                    idx_tensor = torch.LongTensor(range(tfeat_tensor.shape[0])).to(params.device)
                else:
                    idx_tensor = torch.LongTensor(tindex).to(params.device)
                idx_tensor = idx_tensor.clamp(min=0, max=max(0, tfeat_tensor.shape[0]-1))
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=idx_tensor)
                if subFeat_tensor.shape[0] == 0:
                    subFeat_tensor = tfeat_tensor

                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tAA = torch.softmax(tAA, dim=0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
                tPredict = classifier(tattFeat_tensor)
                slide_sub_preds.append(tPredict)

                patch_pred_logits = classifier(tmidFeat)
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

                cls_idx = -1 if patch_pred_softmax.shape[1] > 1 else 0
                if patch_pred_softmax.shape[0] == 0:
                    sort_idx = torch.LongTensor([0]).to(params.device)
                else:
                    _, sort_idx = torch.sort(patch_pred_softmax[:, cls_idx], descending=True)

                k = max(1, instance_per_group)
                if sort_idx.numel() == 0:
                    topk_idx_max = torch.LongTensor([0]).to(params.device)
                    topk_idx_min = torch.LongTensor([0]).to(params.device)
                else:
                    topk_idx_max = sort_idx[:k].long()
                    topk_idx_min = sort_idx[-k:].long()

                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0) if (topk_idx_min.numel() > 0) else topk_idx_max
                topk_idx = topk_idx.clamp(min=0, max=max(0, tmidFeat.shape[0]-1))

                try:
                    MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                except Exception:
                    MaxMin_inst_feat = tattFeat_tensor
                try:
                    max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                except Exception:
                    max_inst_feat = tattFeat_tensor
                af_inst_feat = tattFeat_tensor

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)
                else:
                    slide_pseudo_feat.append(af_inst_feat)

            if len(slide_pseudo_feat) == 0:
                full_tmid = dimReduction(tfeat_tensor)
                full_A = attention(full_tmid).squeeze(0)
                full_tatt = torch.sum(torch.einsum('ns,n->ns', full_tmid, torch.softmax(full_A, dim=0)), dim=0).unsqueeze(0)
                slide_pseudo_feat = [full_tatt]

            try:
                slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
            except Exception:
                slide_pseudo_feat = slide_pseudo_feat[0]

            try:
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
            except Exception:
                agg_att = torch.sum(torch.einsum('ns,n->ns', dimReduction(tfeat_tensor), torch.softmax(attention(dimReduction(tfeat_tensor)).squeeze(0), dim=0)), dim=0).unsqueeze(0)
                slide_sub_preds = classifier(agg_att)
                slide_sub_labels = tslideLabel

            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()

            optimizer0.zero_grad(); optimizer1.zero_grad()
            loss0.backward(retain_graph=True); loss1.backward()

            def _parameters(obj):
                if hasattr(obj, "module"):
                    return obj.module.parameters()
                else:
                    return obj.parameters()

            torch.nn.utils.clip_grad_norm_(_parameters(dimReduction), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(attention), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(classifier), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(UClassifier), params.grad_clipping)

            optimizer0.step(); optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = f'epoch: {epoch} idx: {idx} First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg}'
            try: print_log(tstr, f_log)
            except Exception: print(tstr)

    try:
        if writer is not None:
            writer.add_scalar('train_loss_0', Train_Loss0.avg, epoch)
            writer.add_scalar('train_loss_1', Train_Loss1.avg, epoch)
    except Exception:
        pass

    return Train_Loss0.avg, Train_Loss1.avg

# -------------------------
# CORRECTED main()
# returns: best_auc, test_auc, best_epoch, test_f1, test_acc, val_acc
# -------------------------
def main(params):
    print('will save model' if params.isSaveModel else 'will not save model')

    epoch_step = json.loads(params.epoch_step)
    params.log_dir = os.path.join(params.log_dir, params.p_name)
    writer = SummaryWriter(os.path.join(params.log_dir, params.name))
    log_dir = os.path.join(params.log_dir, str(params.name))

    in_chn = params.in_chn
    try:
        pt_folder = os.path.join(params.data_dir, 'pt_files')
        if os.path.isdir(pt_folder):
            pt_files = sorted([f for f in os.listdir(pt_folder) if f.endswith('.pt')])
            if len(pt_files) > 0:
                sample_tensor = torch.load(os.path.join(pt_folder, pt_files[0]))
                if hasattr(sample_tensor, "shape") and len(sample_tensor.shape) >= 2:
                    detected = int(sample_tensor.shape[1])
                    if detected != in_chn:
                        print(f"Auto-detected feature dim {detected}")
                        in_chn = detected
                        params.in_chn = detected
    except Exception:
        pass

    if not hasattr(params, "numLayer_Res"):
        params.numLayer_Res = 1

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

    if params.num == 0:
        params.isIHC = False

    if params.isIHC:
        util_dict = {"classifier": classifier, "attention": attention, "dim_reduction": dimReduction, "att_classifier": attCls}
        model_path = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\Model\best_model1.pth"
        embed = torch.load(model_path)
        embed = transform_state_dict(embed, util_dict)
        for k in util_dict:
            util_dict[k].load_state_dict(embed[k], strict=False)
        print("Loaded pretrained")

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        dimReduction = torch.nn.DataParallel(dimReduction)

    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)
    os.makedirs(params.log_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(params.log_dir, str(params.name), f"log_{params.num}.txt")
    save_dir = os.path.join(params.log_dir, str(params.name), "best_model")
    with open(log_path, "a") as f:
        f.write(json.dumps(vars(params)))
    log_file = open(log_path, "a")

    SlideNames_train, Label_train = reOrganize_mDATA(params.dataset_csv, params.fold_csv, "train", params.label_col)
    SlideNames_val, Label_val = reOrganize_mDATA(params.dataset_csv, params.fold_csv, "val", params.label_col)
    SlideNames_test, Label_test = reOrganize_mDATA(params.dataset_csv, params.fold_csv, "test", params.label_col)

    print_log(f"Training: {len(SlideNames_train)}, Validation: {len(SlideNames_val)}, Test: {len(SlideNames_test)}", log_file)

    trainable = list(classifier.parameters()) + list(attention.parameters()) + list(dimReduction.parameters())
    optimizer_adam0 = torch.optim.Adam(trainable, lr=params.lr, weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    # --------------------------------------------------------------------
    # 🔥 OPTUNA SCHEDULER INTEGRATION
    # --------------------------------------------------------------------
    def build_scheduler(opt, name):
        if name == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=params.step_size, gamma=params.gamma
            )
        elif name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=params.T_max
            )
        elif name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=params.patience, factor=params.gamma
            )
        else:
            return None  # no scheduler

    scheduler0 = build_scheduler(optimizer_adam0, params.scheduler)
    scheduler1 = build_scheduler(optimizer_adam1, params.scheduler)

    early_stopping = EarlyStopping(patience=20, stop_epoch=85)

    best_auc = 0; best_epoch = -1; test_auc = 0; test_f1 = 0
    final_preds = None; final_gts = None; val_acc = 0.0

    for ii in range(params.EPOCH):
        torch.cuda.empty_cache()
        seed_torch(1)

        train_attention_preFeature_DTFD(
            mDATA_list=(SlideNames_train, Label_train),
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            optimizer0=optimizer_adam0,
            optimizer1=optimizer_adam1,
            epoch=ii,
            ce_cri=ce_cri,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup,
            total_instance=params.total_instance,
            distill=params.distill_type
        )

        # VALIDATION
        print("----- VALIDATION METRICS -----")
        auc_val, f1_val, val_preds, val_gts = test_attention_DTFD_preFeat_MultipleMean(
            mDATA_list=(SlideNames_val, Label_val),
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            epoch=ii,
            criterion=ce_cri,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup_test,
            total_instance=params.total_instance_test,
            distill=params.distill_type
        )
        print(f"[VAL] AUC={auc_val:.4f}, F1={f1_val:.4f}")

        # compute val_acc
        try:
            preds_tensor = val_preds[1]
            gts_tensor = val_gts[1]
            vp = preds_tensor.detach().cpu().numpy()
            vg = gts_tensor.detach().cpu().numpy()
            vp = vp.argmax(axis=1) if vp.ndim == 2 else vp.ravel().astype(int)
            vg = vg.ravel().astype(int)
            val_acc = float(np.mean(vp == vg))
        except:
            val_acc = 0.0

        # TEST
        print("----- TEST METRICS -----")
        tauc, tf1, preds, gts = test_attention_DTFD_preFeat_MultipleMean(
            mDATA_list=(SlideNames_test, Label_test),
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            epoch=ii,
            criterion=ce_cri,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup_test,
            total_instance=params.total_instance_test,
            distill=params.distill_type
        )
        print(f"[EPOCH {ii}] TEST: AUC={tauc:.4f}, F1={tf1:.4f}")

        final_preds = preds; final_gts = gts

        # Save checkpoint for resume
        try:
            torch.save({
                'epoch': ii,
                'optimizer_adam0': optimizer_adam0.state_dict(),
                'optimizer_adam1': optimizer_adam1.state_dict(),
                'classifier': classifier.state_dict(),
                'dim_reduction': dimReduction.state_dict(),
                'attention': attention.state_dict(),
                'att_classifier': attCls.state_dict()
            }, save_dir + "_for_resume.pth")
        except:
            pass

        # Save best by validation AUC
        if auc_val > best_auc:
            best_auc = auc_val; best_epoch = ii; test_auc = tauc; test_f1 = tf1
            if params.isSaveModel:
                try:
                    torch.save({
                        'epoch': ii, 'preds': preds, 'gts': gts,
                        'classifier': classifier.state_dict(),
                        'dim_reduction': dimReduction.state_dict(),
                        'attention': attention.state_dict(),
                        'att_classifier': attCls.state_dict()
                    }, save_dir + ".pth")
                except:
                    pass

        # ----------------------------------------------
        # 🔥 LR SCHEDULER STEP
        # ----------------------------------------------
        if params.scheduler == "ReduceLROnPlateau":
            scheduler0.step(auc_val)
            scheduler1.step(auc_val)
        elif scheduler0 is not None:
            scheduler0.step()
            scheduler1.step()

        if early_stopping(auc_val, ii):
            break

    # compute final test_acc
    test_acc = 0.0
    try:
        preds_tensor = final_preds[1]
        gts_tensor = final_gts[1]
        vp = preds_tensor.detach().cpu().numpy()
        vg = gts_tensor.detach().cpu().numpy()
        vp = vp.argmax(axis=1) if vp.ndim == 2 else vp.ravel().astype(int)
        vg = vg.ravel().astype(int)
        test_acc = float(np.mean(vp == vg))
    except:
        test_acc = 0.0

    try: log_file.close()
    except: pass
    try: writer.close()
    except: pass

    return best_auc, test_auc, best_epoch, test_f1, test_acc, val_acc

# allow running main directly
if __name__ == "__main__":
    seed_torch(1)
    params = parser.parse_args()
    params.p_name = params.name
    if not os.path.isdir(params.log_dir): os.makedirs(params.log_dir, exist_ok=True)
    params.fold_csv = os.path.join(params.splits_dir, 'splits.csv') if params.splits_dir else params.splits_dir
    params.num = getattr(params, 'num', 0)
    main(params)
