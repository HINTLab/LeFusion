from monai.metrics import compute_surface_dice
import nibabel as nib
import numpy as np
import torch

def nsd(predict_path, label_path, space_path, tolerance=[1.0]):

    predict = nib.load(predict_path)
    label = nib.load(label_path)
    space = nib.load(space_path)
    
    affine = space.affine
    spacing = affine[:3, :3].diagonal()
    spacing = np.abs(spacing)

    pred = predict.get_fdata()
    lal = label.get_fdata()
    pred = torch.tensor(pred).int()
    pred = pred.unsqueeze(1)

    surface_dice = compute_surface_dice(y_pred=pred, 
                                        y=lal, 
                                        spacing=spacing, 
                                        class_thresholds=tolerance)
    return surface_dice
