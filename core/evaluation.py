# import torch
#
# def get_curve_online_torch(known, novel, stypes=['Bas']):
#     tp, fp = {}, {}
#     tnr_at_tpr95 = {}
#
#     for stype in stypes:
#         known, _ = torch.sort(known)
#         novel, _ = torch.sort(novel)
#
#         num_k = known.shape[0]
#         num_n = novel.shape[0]
#
#         tp[stype] = -torch.ones(num_k + num_n + 1, dtype=torch.int32, device=known.device)
#         fp[stype] = -torch.ones(num_k + num_n + 1, dtype=torch.int32, device=novel.device)
#
#         tp[stype][0] = num_k
#         fp[stype][0] = num_n
#
#         k, n = 0, 0
#         for l in range(num_k + num_n):
#             if k == num_k:
#                 tp[stype][l+1:] = tp[stype][l]
#                 fp[stype][l+1:] = torch.arange(fp[stype][l]-1, -1, -1, device=fp[stype].device)
#                 break
#             elif n == num_n:
#                 tp[stype][l+1:] = torch.arange(tp[stype][l]-1, -1, -1, device=tp[stype].device)
#                 fp[stype][l+1:] = fp[stype][l]
#                 break
#             else:
#                 if novel[n] < known[k]:
#                     n += 1
#                     tp[stype][l+1] = tp[stype][l]
#                     fp[stype][l+1] = fp[stype][l] - 1
#                 else:
#                     k += 1
#                     tp[stype][l+1] = tp[stype][l] - 1
#                     fp[stype][l+1] = fp[stype][l]
#
#         tpr95_pos = torch.abs(tp[stype].float() / num_k - 0.95).argmin()
#         tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos].float() / num_n
#
#     return tp, fp, tnr_at_tpr95
#
#
# def metric_ood(x1, x2, stypes=['Bas'], verbose=True):
#     # 如果已经是 Tensor（通常是），那就直接用：
#     x1 = x1.to('cuda') if torch.cuda.is_available() else x1.to('cpu')
#     x2 = x2.to(x1.device)
#
#     tp, fp, tnr_at_tpr95 = get_curve_online_torch(x1, x2, stypes)
#
#     results = {}
#     for stype in stypes:
#         results[stype] = {}
#
#         num_k = tp[stype][0].item()
#         num_n = fp[stype][0].item()
#
#         tpr = torch.cat([torch.tensor([1.], device=x1.device), tp[stype].float() / num_k, torch.tensor([0.], device=x1.device)])
#         fpr = torch.cat([torch.tensor([1.], device=x1.device), fp[stype].float() / num_n, torch.tensor([0.], device=x1.device)])
#
#         results[stype]['TNR'] = 100. * tnr_at_tpr95[stype].item()
#         results[stype]['AUROC'] = 100. * (-torch.trapz(1. - fpr, tpr)).item()
#         results[stype]['DTACC'] = 100. * (0.5 * (tp[stype].float() / num_k + 1. - fp[stype].float() / num_n).max()).item()
#
#         denom_in = tp[stype] + fp[stype]
#         denom_in = denom_in.float()
#         denom_in[denom_in == 0.] = -1.
#         pin = tp[stype].float() / denom_in
#         pin = torch.cat([torch.tensor([0.5], device=x1.device), pin, torch.tensor([0.], device=x1.device)])
#         pin_ind = (denom_in > 0.)
#         pin_ind = torch.cat([torch.tensor([True], device=x1.device), pin_ind, torch.tensor([True], device=x1.device)])
#         results[stype]['AUIN'] = 100. * (-torch.trapz(pin[pin_ind], tpr[pin_ind])).item()
#
#         denom_out = num_k - tp[stype] + num_n - fp[stype]
#         denom_out = denom_out.float()
#         denom_out[denom_out == 0.] = -1.
#         pout = (num_n - fp[stype].float()) / denom_out
#         pout = torch.cat([torch.tensor([0.], device=x1.device), pout, torch.tensor([0.5], device=x1.device)])
#         pout_ind = (denom_out > 0.)
#         pout_ind = torch.cat([torch.tensor([True], device=x1.device), pout_ind, torch.tensor([True], device=x1.device)])
#         results[stype]['AUOUT'] = 100. * (torch.trapz(pout[pout_ind], 1. - fpr[pout_ind])).item()
#
#         if verbose:
#             print(f"{stype:5s} TNR {results[stype]['TNR']:.3f} AUROC {results[stype]['AUROC']:.3f} "
#                   f"DTACC {results[stype]['DTACC']:.3f} AUIN {results[stype]['AUIN']:.3f} "
#                   f"AUOUT {results[stype]['AUOUT']:.3f}")
#
#     return results
#
#
# def compute_oscr(pred_k, pred_u, labels):
#     device = pred_k.device if isinstance(pred_k, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     pred_k = pred_k if isinstance(pred_k, torch.Tensor) else torch.tensor(pred_k, device=device)
#     pred_u = pred_u if isinstance(pred_u, torch.Tensor) else torch.tensor(pred_u, device=device)
#     labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
#
#     x1 = torch.max(pred_k, dim=1)[0]
#     x2 = torch.max(pred_u, dim=1)[0]
#     pred = torch.argmax(pred_k, dim=1)
#
#     correct = (pred == labels)
#     m_x1 = torch.zeros_like(x1)
#     m_x1[correct] = 1
#
#     k_target = torch.cat([m_x1, torch.zeros_like(x2)], dim=0)
#     u_target = torch.cat([torch.zeros_like(x1), torch.ones_like(x2)], dim=0)
#     predict = torch.cat([x1, x2], dim=0)
#
#     n = predict.size(0)
#     CCR = torch.zeros(n + 2, device=device)
#     FPR = torch.zeros(n + 2, device=device)
#
#     idx = predict.argsort()
#     s_k_target = k_target[idx]
#     s_u_target = u_target[idx]
#
#     for k in range(n - 1):
#         CCR[k] = s_k_target[k+1:].sum() / x1.size(0)
#         FPR[k] = s_u_target[k:].sum() / x2.size(0)
#
#     CCR[n] = 0.0
#     FPR[n] = 0.0
#     CCR[n+1] = 1.0
#     FPR[n+1] = 1.0
#
#     ROC = sorted(zip(FPR.tolist(), CCR.tolist()), reverse=True)
#
#     OSCR = 0.0
#     for j in range(n + 1):
#         h = ROC[j][0] - ROC[j+1][0]
#         w = (ROC[j][1] + ROC[j+1][1]) / 2.0
#         OSCR += h * w
#
#     return OSCR

import os
import sys
import numpy as np


def get_curve_online(known, novel, stypes=['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes=['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100. * tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = 100. * (-np.trapz(1. - fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100. * (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[stype][mtype] = 100. * (-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[stype][mtype] = 100. * (np.trapz(pout[pout_ind], 1. - fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')

    return results


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR