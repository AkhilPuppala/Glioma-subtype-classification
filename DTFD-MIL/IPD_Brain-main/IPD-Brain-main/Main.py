import time
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import random
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    # TensorBoard not available in this environment; provide a dummy writer
    class SummaryWriter:
        def init(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_image(self, *args, **kwargs):
            pass
        def close(self):
            pass
import pickle
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric, eval_metric_
import pandas as pd

parser = argparse.ArgumentParser(description='abc')
testMask_dir = '' ## Point to the Camelyon test set mask location

parser.add_argument('--name', default='abc', type=str)

#### IHC 
parser.add_argument('--label_col', default='label', type=str)
parser.add_argument('--isIHC', default=False, type=bool)
#####

parser.add_argument('--k_start', default=-1, type=int)
parser.add_argument('--k_end', default=-1, type=int)

parser.add_argument('--isPar', default=True, type=bool)

parser.add_argument('--splits_dir', default='', type=str)  ## Dataset_csv
parser.add_argument('--dataset_csv', default='', type=str)  ## Dataset_csv

parser.add_argument('--num_cls', default=3, type=int)
parser.add_argument('--data_dir', default='/scratch/ekansh.chauhan/FEATURES_DIRECTORY', type=str)  ## feature_dir

parser.add_argument('--in_chn', default=384, type=int)

parser.add_argument('--mDim', default=384, type=int)

parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[90]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_dir', default='./results', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=200, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
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
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(params):
    if params.isSaveModel:
        print('will save model')
    else:
        print('will not save model')

    epoch_step = json.loads(params.epoch_step)
    params.log_dir = os.path.join(params.log_dir, params.p_name)
    writer = SummaryWriter(os.path.join(params.log_dir, params.name))
    log_dir = os.path.join(params.log_dir, str(params.name))

    in_chn = params.in_chn

    # --- Auto-detect input feature dimension if .pt files available ---
    try:
        pt_folder = os.path.join(params.data_dir, 'pt_files')
        if os.path.isdir(pt_folder):
            pt_files = sorted([f for f in os.listdir(pt_folder) if f.endswith('.pt')])
            if len(pt_files) > 0:
                sample_path = os.path.join(pt_folder, pt_files[0])
                sample_tensor = torch.load(sample_path)
                if hasattr(sample_tensor, 'shape') and len(sample_tensor.shape) >= 2:
                    detected_in_chn = int(sample_tensor.shape[1])
                    if detected_in_chn != in_chn:
                        print(f'Auto-detected feature dim {detected_in_chn} from {pt_files[0]}, overriding --in_chn {in_chn} -> {detected_in_chn}')
                        in_chn = detected_in_chn
                        params.in_chn = detected_in_chn
    except Exception as e:
        print("Feature dimension auto-detection skipped:", e)

    # --- Model initialization ---
    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

    if params.num == 0:
        params.isIHC = False

    # --- Load pretrained model if IHC enabled ---
    if params.isIHC:
        util_dict = {"classifier": classifier, "attention": attention, "dim_reduction": dimReduction, "att_classifier": attCls}
        if params.num != 0:
            molecular_model_path = os.path.join(r'D:\IPD\IPD-Brain-main\IPD-Brain-main\Model', 'best_model1.pth')

        embed = torch.load(molecular_model_path)
        embed = transform_state_dict(embed, util_dict)

        for i in util_dict.keys():
            w_to_use = embed[i]
            util_dict[i].load_state_dict(w_to_use, strict=False)
        print("Loaded pre-trained model")

    # --- Parallelization ---
    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        dimReduction = torch.nn.DataParallel(dimReduction)

    # --- Loss and Optimizers ---
    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    os.makedirs(params.log_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(params.log_dir, str(params.name), f'log_{params.num}.txt')
    save_dir = os.path.join(params.log_dir, str(params.name), 'best_model')
    z = vars(params).copy()

    with open(log_path, 'a') as f:
        f.write(json.dumps(z))

    log_file = open(log_path, 'a')

    # --- Dataset splits ---
    SlideNames_train, Label_train = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'train', label_name=params.label_col)
    SlideNames_val, Label_val = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'val', label_name=params.label_col)
    SlideNames_test, Label_test = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'test', label_name=params.label_col)
    print(params.name, params.fold_csv)

    print_log(f'Folder name: {params.name}, Fold: {params.num + 1}', log_file)
    print_log(f'Training: {len(SlideNames_train)}, Validation: {len(SlideNames_val)}, Test: {len(SlideNames_test)}', log_file)

    # --- Optimizers & schedulers ---
    trainable_parameters = list(classifier.parameters()) + list(attention.parameters()) + list(dimReduction.parameters())
    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr, weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    # --- Early Stopping setup ---
    early_stopping = EarlyStopping(patience=20, stop_epoch=85)

    best_auc = 0
    best_epoch = -1
    test_auc = 0
    test_f1 = 0

    # --- Training Loop ---
    for ii in range(params.EPOCH):
        torch.cuda.empty_cache()
        seed_torch(seed=1)

        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f'Current LR: {curLR}', log_file)

        start_time = time.time()

        # --- Training step ---
        train_attention_preFeature_DTFD(
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            mDATA_list=(SlideNames_train, Label_train),
            ce_cri=ce_cri,
            optimizer0=optimizer_adam0,
            optimizer1=optimizer_adam1,
            epoch=ii,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup,
            total_instance=params.total_instance,
            distill=params.distill_type
        )

        print(f'Epoch {ii} training time: {time.time()-start_time:.2f}s')

        # --- Validation ---
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
        auc_val, f1_val, _, _ = test_attention_DTFD_preFeat_MultipleMean(
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            mDATA_list=(SlideNames_val, Label_val),
            criterion=ce_cri,
            epoch=ii,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup_test,
            total_instance=params.total_instance_test,
            distill=params.distill_type
        )

        # --- Test evaluation ---
        print_log(f'>>>>>>>>>>> Test Epoch: {ii}', log_file)
        tauc, tf1, preds, gts = test_attention_DTFD_preFeat_MultipleMean(
            classifier=classifier,
            dimReduction=dimReduction,
            attention=attention,
            UClassifier=attCls,
            mDATA_list=(SlideNames_test, Label_test),
            criterion=ce_cri,
            epoch=ii,
            params=params,
            f_log=log_file,
            writer=writer,
            numGroup=params.numGroup_test,
            total_instance=params.total_instance_test,
            distill=params.distill_type
        )

        # --- Save checkpoint for resume ---
        resume_ckpt = {
            'epoch': ii,
            'optimizer_adam0': optimizer_adam0.state_dict(),
            'optimizer_adam1': optimizer_adam1.state_dict(),
            'classifier': classifier.state_dict(),
            'dim_reduction': dimReduction.state_dict(),
            'attention': attention.state_dict(),
            'att_classifier': attCls.state_dict()
        }
        torch.save(resume_ckpt, save_dir + '_for_resume.pth')

        # --- Save best model ---
        if auc_val > best_auc:
            best_auc = auc_val
            best_epoch = ii
            test_auc = tauc
            test_f1 = tf1

            if params.isSaveModel:
                print('Saving best model...')
                best_ckpt = {
                    'epoch': ii,
                    'preds': preds,
                    'gts': gts,
                    'classifier': classifier.state_dict(),
                    'dim_reduction': dimReduction.state_dict(),
                    'attention': attention.state_dict(),
                    'att_classifier': attCls.state_dict()
                }
                torch.save(best_ckpt, save_dir + '.pth')
                alt_save_path = r'D:\IPD\IPD-Brain-main\IPD-Brain-main\Model\best_model1.pth'
                torch.save(best_ckpt, alt_save_path)
                print(f"\n✅ Model saved at:\n  - {save_dir}.pth\n  - {alt_save_path}")

        print_log(f'Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f} (Best Epoch: {best_epoch})', log_file)

        # --- Scheduler step ---
        scheduler0.step()
        scheduler1.step()

        # --- Early stopping check (AUC-based) ---
        if early_stopping(auc_val, ii):
            print_log(f"\n⏹ Early stopping triggered at epoch {ii} (best AUC {best_auc:.4f})", log_file)
            break

    return best_auc, test_auc, best_epoch, f1_val



def transform_state_dict(state_dict, util_dict):
    for j in util_dict.keys():
        var = state_dict[j]
        new_state_dict = {}

        for k, v in var.items():
            name = k[7:]
            if k[:6] == "module":
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        keys_list = list(new_state_dict.keys())

        for i in keys_list:
            if new_state_dict[i].shape[0] == 3:
                new_state_dict[i+'_not_to_use'] = new_state_dict[i]
                del new_state_dict[i]
        state_dict[j] = new_state_dict
    return state_dict


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, 
                                             criterion=None,  params=None, f_log=None, writer=None, numGroup=3, 
                                             total_instance=3, distill='MaxMinS'):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    sl_names = []
    SlideNames, Label = mDATA_list
    # ensure at least 1 instance per group
    instance_per_group = max(1, total_instance // max(1, numGroup))

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        # handle case where batch_size_v might not divide numSlides exactly
        numIter = max(1, (numSlides + params.batch_size_v - 1) // params.batch_size_v)
        tIDX = list(range(numSlides))

        for idx in range(numIter):
            
            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            if len(tidx_slide) == 0:
                continue

            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)

            # load features for batch slides
            batch_feat = []
            for sst in tidx_slide:
                pt_path = os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(SlideNames[sst]))
                try:
                    loaded = torch.load(pt_path)
                except Exception as e:
                    print(f'Error loading {pt_path}: {e}')
                    # create a dummy minimal tensor to avoid crash but report error
                    loaded = torch.zeros((1, params.in_chn))
                loaded = loaded.to(params.device)
                batch_feat.append(loaded)

            # fix sl_names append bug: extend with list of names
            sl_names.extend([SlideNames[sst] for sst in tidx_slide])

            # iterate slides in batch
            for tidx, tfeat in enumerate(batch_feat):
                if tfeat is None or tfeat.numel() == 0:
                    continue

                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)
                
                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []
                allSlide_pred_logits = []

                for jj in range(max(1, params.num_MeanInference)):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    # remove empty groups
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list if len(sst) > 0]

                    if len(index_chunk_list) == 0:
                        # fallback: use all indices as single group
                        index_chunk_list = [feat_index]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)

                        # ensure index tensor is not empty
                        if len(tindex) == 0:
                            continue

                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        # protect in case idx_tensor larger than available
                        idx_tensor = idx_tensor.clamp(min=0, max=max(0, midFeat.shape[0]-1))
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        # if tmidFeat has zero rows after selection, skip
                        if tmidFeat.shape[0] == 0:
                            continue

                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x C
                        slide_sub_preds.append(tPredict)

                        # patch-level logits/probs for selection
                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        # if classes are single-dim, ensure indexing doesn't crash
                        cls_idx = -1 if patch_pred_softmax.shape[1] > 1 else 0
                        _, sort_idx = torch.sort(patch_pred_softmax[:, cls_idx], descending=True)

                        # ensure at least 1 instance selected
                        k = max(1, instance_per_group)
                        topk_idx_max = sort_idx[:k].long() if sort_idx.numel() > 0 else torch.LongTensor([]).to(params.device)
                        topk_idx_min = sort_idx[-k:].long() if sort_idx.numel() > 0 else torch.LongTensor([]).to(params.device)

                        if distill == 'MaxMinS':
                            # combine unique indices (avoid duplicates)
                            if topk_idx_max.numel() == 0 and topk_idx_min.numel() == 0:
                                # fallback: select the top 1 index (if available)
                                if sort_idx.numel() > 0:
                                    topk_idx_max = sort_idx[:1].long()
                                else:
                                    topk_idx_max = torch.LongTensor([]).to(params.device)
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0) if topk_idx_min.numel()>0 else topk_idx_max
                            if topk_idx.numel() == 0:
                                # fallback to tattFeat_tensor if no valid topk
                                d_inst_feat = tattFeat_tensor
                            else:
                                # ensure indices are within range for tmidFeat
                                topk_idx = topk_idx.clamp(min=0, max=max(0, tmidFeat.shape[0]-1))
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx.to(tmidFeat.device))
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            if topk_idx_max.numel() == 0:
                                if sort_idx.numel() > 0:
                                    topk_idx_max = sort_idx[:1].long()
                                else:
                                    topk_idx_max = torch.LongTensor([]).to(params.device)
                            if topk_idx_max.numel() == 0:
                                d_inst_feat = tattFeat_tensor
                            else:
                                topk_idx_max = topk_idx_max.clamp(min=0, max=max(0, tmidFeat.shape[0]-1))
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max.to(tmidFeat.device))
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    # if no slide_d_feat collected, fallback to aggregated tattFeat_tensor of full midFeat
                    if len(slide_d_feat) == 0:
                        # compute a fallback: attention pooling over whole slide
                        tAA_full = torch.softmax(AA, dim=0)
                        tattFeats_full = torch.einsum('ns,n->ns', midFeat, tAA_full)
                        tattFeat_tensor_full = torch.sum(tattFeats_full, dim=0).unsqueeze(0)
                        slide_d_feat = [tattFeat_tensor_full]

                    try:
                        slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    except Exception:
                        # if sizes mismatch, stack by padding or take first
                        slide_d_feat = slide_d_feat[0]

                    # prepare slide_sub_preds/labels
                    if len(slide_sub_preds) == 0:
                        # fallback prediction from aggregated attention feature
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

                    gSlidePred = UClassifier(slide_d_feat)  # logits
                    allSlide_pred_logits.append(gSlidePred)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                # compute mean logits (preferred for loss)
                if len(allSlide_pred_logits) == 0:
                    # fallback: use avg of softmax (as logits unknown)
                    allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0) if len(allSlide_pred_softmax)>0 else torch.zeros((1, params.num_cls)).to(params.device)
                    allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                    # convert to logits by log (with clamp)
                    mean_logits = torch.log(allSlide_pred_softmax + 1e-12)
                else:
                    mean_logits = torch.mean(torch.stack(allSlide_pred_logits, dim=0), dim=0)  # average logits
                    # ensure shape is [1, C]
                    if mean_logits.dim() == 1:
                        mean_logits = mean_logits.unsqueeze(0)

                # now compute loss using logits
                try:
                    loss1 = F.cross_entropy(mean_logits, tslideLabel)
                except Exception:
                    # fallback to using softmax->nll if cross_entropy fails
                    mean_soft = torch.softmax(mean_logits, dim=1)
                    loss1 = F.nll_loss(torch.log(mean_soft + 1e-12), tslideLabel)

                test_loss1.update(loss1.item(), 1)

                # record predicted probabilities for metrics
                probs = torch.softmax(mean_logits, dim=1)
                gPred_1 = torch.cat([gPred_1, probs], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

    # convert first-tier preds to probs
    if gPred_0.numel() > 0:
        gPred_0 = torch.softmax(gPred_0, dim=1)
    else:
        gPred_0 = gPred_0

    # evaluate metrics
    if params.num_cls == 2:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)
    else:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric_(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric_(gPred_1, gt_1)

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    # writer scalars
    try:
        writer.add_scalar(f'auc_0 ', auc_0, epoch)
        writer.add_scalar(f'auc_1 ', auc_1, epoch)
        writer.add_scalar(f'F1_0 ', mF1_0, epoch)
        writer.add_scalar(f'F1_1 ', mF1_1, epoch)
        writer.add_scalar(f'Acc_0 ', macc_0, epoch)
        writer.add_scalar(f'Acc_1 ', macc_1, epoch)
    except Exception:
        pass

    return auc_1, mF1_1, (gPred_0, gPred_1), (gt_0, gt_1)



def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, 
                                    epoch, ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=3, 
                                    distill='MaxMinS'):

    SlideNames_list, Label_dict = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    # ensure at least 1 instance per group
    instance_per_group = max(1, total_instance // max(1, numGroup))

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    # handle case when batch_size doesn't divide exactly
    numIter = max(1, (numSlides + params.batch_size - 1) // params.batch_size)

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]
        if len(tidx_slide) == 0:
            continue

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_path = os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(tslide))
            try:
                tfeat_tensor = torch.load(tfeat_path)
            except Exception as e:
                print(f'Warning: failed to load {tfeat_path}: {e}')
                tfeat_tensor = torch.zeros((1, params.in_chn))
            # ensure float tensor on correct device
            if not isinstance(tfeat_tensor, torch.Tensor):
                tfeat_tensor = torch.tensor(tfeat_tensor, dtype=torch.float32)
            tfeat_tensor = tfeat_tensor.to(params.device)

            # build indices and chunks
            feat_count = max(1, tfeat_tensor.shape[0])
            feat_index = list(range(feat_count))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            # remove empty chunks
            index_chunk_list = [sst.tolist() for sst in index_chunk_list if len(sst) > 0]

            if len(index_chunk_list) == 0:
                index_chunk_list = [feat_index]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)

                if len(tindex) == 0:
                    idx_tensor = torch.LongTensor(range(tfeat_tensor.shape[0])).to(params.device)
                else:
                    idx_tensor = torch.LongTensor(tindex).to(params.device)

                # guard index tensor bounds
                idx_tensor = idx_tensor.clamp(min=0, max=max(0, tfeat_tensor.shape[0]-1))

                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=idx_tensor)
                if subFeat_tensor.shape[0] == 0:
                    subFeat_tensor = tfeat_tensor

                # Forward pass for this chunk
                tmidFeat = dimReduction(subFeat_tensor)
                # attention expects shape [1, N, feat] or [N, feat] depending on implementation; using as in your code
                tAA = attention(tmidFeat).squeeze(0)
                tAA = torch.softmax(tAA, dim=0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x num_cls
                slide_sub_preds.append(tPredict)

                # patch-level logits/probs for selection
                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                cls_idx = -1 if patch_pred_softmax.shape[1] > 1 else 0
                if patch_pred_softmax.shape[0] == 0:
                    # fallback to index 0
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

                # extract features for distillation variants
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
                    # unknown distill mode -> fallback to aggregated attention feature
                    slide_pseudo_feat.append(af_inst_feat)

            # ensure slide_pseudo_feat not empty
            if len(slide_pseudo_feat) == 0:
                # fallback to aggregated attention over the whole slide
                full_tmid = dimReduction(tfeat_tensor)
                full_A = attention(full_tmid).squeeze(0)
                full_tatt = torch.sum(torch.einsum('ns,n->ns', full_tmid, torch.softmax(full_A, dim=0)), dim=0).unsqueeze(0)
                slide_pseudo_feat = [full_tatt]

            # try to concat; if shapes differ, keep them as-is (first element)
            try:
                slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
            except Exception:
                slide_pseudo_feat = slide_pseudo_feat[0]

            # prepare first-tier predictions & losses
            try:
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x C
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            except Exception:
                # if something failed, fallback to single aggregated prediction
                agg_att = torch.sum(torch.einsum('ns,n->ns', dimReduction(tfeat_tensor), torch.softmax(attention(dimReduction(tfeat_tensor)).squeeze(0), dim=0)), dim=0).unsqueeze(0)
                slide_sub_preds = classifier(agg_att)
                slide_sub_labels = tslideLabel

            # compute losses
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()

            # Zero grads for both optimizers before backward
            optimizer0.zero_grad()
            optimizer1.zero_grad()

            # Backprop both losses; retain_graph=True for first to allow second backward if they share graph
            # If graphs are independent, PyTorch will not require retain_graph; using True to be safe with shared parts.
            loss0.backward(retain_graph=True)
            loss1.backward()

            # Gradient clipping: supports DataParallel wrappers
            def _parameters(obj):
                # helper to obtain parameters whether model is DataParallel or plain
                if hasattr(obj, "module"):
                    return obj.module.parameters()
                else:
                    return obj.parameters()

            torch.nn.utils.clip_grad_norm_(_parameters(dimReduction), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(attention), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(classifier), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(_parameters(UClassifier), params.grad_clipping)

            optimizer0.step()
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        # periodic logging
        if idx % params.train_show_freq == 0:
            tstr = f'epoch: {epoch} idx: {idx} First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg}'
            try:
                print_log(tstr, f_log)
            except Exception:
                print(tstr)

    # write to tensorboard if available
    try:
        if writer is not None:
            writer.add_scalar('train_loss_0', Train_Loss0.avg, epoch)
            writer.add_scalar('train_loss_1', Train_Loss1.avg, epoch)
    except Exception:
        pass

    # return averages (optional)
    return Train_Loss0.avg, Train_Loss1.avg

class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=85):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_metric = float('-inf')
        self.early_stop = False

    def __call__(self, val_metric, epoch):
        # Do not start early stopping before stop_epoch
        if epoch < self.stop_epoch:
            return False
        
        # If performance improves
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


    def _call_(self, val_metric, epoch):
        if epoch < self.stop_epoch:
            return False
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA):

    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(dataset_csv, fold_csv, set_type, label_name='label'):

    SlideNames = []
    Label = []

    mDATA_slides = pd.read_csv(fold_csv)
    mDATA_label = pd.read_csv(dataset_csv)

    # temp_SlideNames should be a column in the splits CSV containing slide identifiers for the set_type
    if set_type not in mDATA_slides.columns:
        raise KeyError(f"Expected column '{set_type}' in splits CSV ({fold_csv}). Available columns: {list(mDATA_slides.columns)}")

    temp_SlideNames = mDATA_slides[set_type].dropna().tolist()

    # dataset CSV may use a different column name for slide id. try common alternatives
    possible_slide_cols = ['slide_id', 'SlideID', 'Slide_Id', 'slide', 'Slide', 'filename', 'file', 'image']
    slide_col = None
    for col in possible_slide_cols:
        if col in mDATA_label.columns:
            slide_col = col
            break

    if slide_col is None:
        raise KeyError(f"Could not find a slide id column in dataset CSV ({dataset_csv}). Tried: {possible_slide_cols}. Available columns: {list(mDATA_label.columns)}")

    # Filter rows that belong to the requested split
    mDATA = mDATA_label[mDATA_label[slide_col].isin(temp_SlideNames)]

    # If label column missing, raise informative error
    if label_name not in mDATA.columns:
        raise KeyError(f"Label column '{label_name}' not found in dataset CSV ({dataset_csv}). Available columns: {list(mDATA.columns)}")

    mapping = {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2}
    # If label column contains strings, map unique string labels to integer classes
    if mDATA[label_name].dtype == object:
        # --- Ensure mDATA is a safe copy ---
        mDATA = mDATA.copy()

        # --- Try known mapping first ---
        if set(mDATA[label_name].unique()).issubset(set(mapping.keys())):
            mDATA = mDATA.replace({label_name: mapping})
            print(f"Applied predefined mapping for labels: {mapping}")
        else:
            # --- Factorize to ints (consistent, deterministic order) ---
            unique_labels = sorted(mDATA[label_name].unique())  # sort ensures fixed order
            label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

            # --- Assign using .loc to avoid SettingWithCopyWarning ---
            mDATA.loc[:, label_name] = mDATA[label_name].map(label_to_int)

            print(f"Auto-mapped label values to integers (fixed): {label_to_int}")


    SlideNames = mDATA[slide_col].tolist()
    Label = mDATA[label_name].tolist()

    # If no rows matched, try matching by basename (strip file extensions) as a fallback
    if len(SlideNames) == 0:
        temp_basenames = [os.path.splitext(s)[0] for s in temp_SlideNames]
        mDATA['basename'] = mDATA[slide_col].apply(lambda x: os.path.splitext(str(x))[0])
        mDATA_basename = mDATA[mDATA['basename'].isin(temp_basenames)]
        if len(mDATA_basename) > 0:
            SlideNames = mDATA_basename[slide_col].tolist()
            Label = mDATA_basename[label_name].tolist()

    ## to test
    # print('SlideNames: ', SlideNames, 'Label: ', Label)

    return SlideNames, Label

if __name__ == "__main__": 
    # Set random seed for reproducibility 
    seed_torch(seed=1) 

    # Parse command-line arguments 
    params = parser.parse_args() 
    params.p_name = params.name 

    # Make sure log directory exists 
    if not os.path.isdir(params.log_dir): 
        os.makedirs(params.log_dir) 

    # Directly use your single splits CSV 
    params.fold_csv = os.path.join(params.splits_dir, 'splits.csv') 

    # Set params.num for scripts that rely on fold number 
    params.num = 0 

    # Optionally, set device and number of layers if not passed as arguments 
    if not hasattr(params, 'device'): 
        params.device = 'cpu'  # or 'cuda:0' if GPU is available 
    if not hasattr(params, 'numLayer_Res'): 
        params.numLayer_Res = 50  # ResNet50 

    # Call main function 
    
    main(params)