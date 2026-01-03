import os
import os.path as osp
import numpy as np

import csv

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation

# def test(net, criterion, testloader, outloader, epoch=None, **options):
#     net.eval()
#     correct, total = 0, 0
#
#     torch.cuda.empty_cache()
#
#     _pred_k, _pred_u, _labels = [], [], []
#
#     with torch.no_grad():
#         for data, labels in testloader:
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
#
#             with torch.set_grad_enabled(False):
#                 x, y = net(data, True)
#                 logits, _ = criterion(x, y)
#                 predictions = logits.data.max(1)[1]
#                 total += labels.size(0)
#                 correct += (predictions == labels.data).sum()
#
#                 _pred_k.append(logits.data.cpu().numpy())
#                 _labels.append(labels.data.cpu().numpy())
#
#         for batch_idx, (data, labels) in enumerate(outloader):
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
#
#             with torch.set_grad_enabled(False):
#                 x, y = net(data, True)
#                 # x, y = net(data, return_feature=True)
#                 logits, _ = criterion(x, y)
#                 _pred_u.append(logits.data.cpu().numpy())
#
#     # Accuracy
#     acc = float(correct) * 100. / float(total)
#     print('Acc: {:.5f}'.format(acc))
#
#     _pred_k = np.concatenate(_pred_k, 0)
#     _pred_u = np.concatenate(_pred_u, 0)
#     _labels = np.concatenate(_labels, 0)
#
#     # Out-of-Distribution detction evaluation
#     x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
#     results = evaluation.metric_ood(x1, x2)['Bas']
#
#     # OSCR
#     _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)
#
#     results['ACC'] = acc
#     results['OSCR'] = _oscr_socre * 100.
#
#     return results

def test_cls(net, criterion, testloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            x, y = net(data, True)
            logits, _ = criterion(x, y)
            predictions = logits.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    results = {'ACC': acc}
    return results




def test_per_class(net, criterion, testloader, save_path="per_class_acc.csv", epoch=None, **options):
    net.eval()
    num_classes = len(testloader.dataset.classes) if hasattr(testloader.dataset, "classes") else None

    correct_per_class = {}
    total_per_class = {}

    torch.cuda.empty_cache()

    with torch.no_grad():
        for data, labels in testloader:
            if options.get('use_gpu', False):
                data, labels = data.cuda(), labels.cuda()

            x, y = net(data, True)
            logits, _ = criterion(x, y)
            predictions = logits.data.max(1)[1]

            for label, pred in zip(labels, predictions):
                label = label.item()
                total_per_class[label] = total_per_class.get(label, 0) + 1
                if pred.item() == label:
                    correct_per_class[label] = correct_per_class.get(label, 0) + 1

    
    per_class_acc = {}
    for cls in sorted(total_per_class.keys()):
        acc = 100.0 * correct_per_class.get(cls, 0) / total_per_class[cls]
        per_class_acc[cls] = round(acc, 2) 

   
    print("Per-class Accuracy:")
    for cls, acc in per_class_acc.items():
        cls_name = testloader.dataset.classes[cls] if num_classes is not None else ""
        print(f"Class {cls} ({cls_name}): {acc:.2f}%")

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ClassID", "ClassName", "Accuracy"])
        for cls, acc in per_class_acc.items():
            cls_name = testloader.dataset.classes[cls] if num_classes is not None else ""
            writer.writerow([cls, cls_name, f"{acc:.2f}"])

    print(f"Per-class accuracy saved to {save_path}")

    return per_class_acc
