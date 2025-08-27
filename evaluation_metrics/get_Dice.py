import nibabel as nib
import numpy as np
import torch

def compute_dice(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    predict = preds.reshape(preds.shape[0], -1)
    target = labels.reshape(labels.shape[0], -1)
    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        num = np.sum(np.multiply(predict, target), axis=1)
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)
        dice = 2 * num / den
        return dice.mean()
    
def dice(predict_path, label_path):
        
        predict = nib.load(predict_path)
        label = nib.load(label_path)
        
        pred = predict.get_fdata()
        pred_sta = pred[0]
        lal = label.get_fdata()
        lal_sta = lal[0]
        lal_sta = torch.tensor(lal_sta).long()
        pred_sta = torch.tensor(pred_sta).long()

        dice = compute_dice(pred_sta, lal_sta)

        return dice


