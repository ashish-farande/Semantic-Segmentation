import numpy as np
import torch

def getTrueCounts(input):
    counts = 0
    unique_values, occurrences = torch.unique(input, return_counts=True)
    if True in unique_values:
        counts = occurrences[(unique_values == True).nonzero(as_tuple=True)[0]].item()  # Supposed the occurrences return (true_num, false_num)
    return counts

def iou(pred, target, n_classes=10):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for undefined class ("9")
    for cls in range(n_classes - 1):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = getTrueCounts(torch.logical_and(pred_inds, target_inds))
        union = getTrueCounts(torch.logical_or(pred_inds, target_inds))
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection/union)
    # ious /= (n_classes - 1)
    return np.array(ious)

def pixel_acc(pred, target):
    # TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    pred = pred.view(-1)
    target = target.view(-1)
    correct = torch.eq(pred, target)
    defined_targets = torch.logical_not(target == 9)
    defined_correct = getTrueCounts(torch.logical_and(correct, defined_targets))
    size = getTrueCounts(defined_targets)
    acc = defined_correct/size
    return acc
