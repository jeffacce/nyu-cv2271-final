import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm


def calc_confusion(pred_bool, target_bool):
    not_pred_bool = ~pred_bool
    not_target_bool = ~target_bool
    tp = (pred_bool & target_bool).sum().item()
    tn = (not_pred_bool & not_target_bool).sum().item()
    fp = (pred_bool & not_target_bool).sum().item()
    fn = (not_pred_bool & target_bool).sum().item()
    return tp, tn, fp, fn


def calc_fp_sensitivity(pred_bool, target_bool):
    tp, tn, fp, fn = calc_confusion(pred_bool, target_bool)
    # sensitivity (recall)
    sensitivity = tp / (tp + fn)
    return fp, sensitivity


# yhat is a float probability in (0, 1) after sigmoid
# target_bool is True/False
# fp_rates is the specified false positive rates per scan
def calc_froc(yhat, target_bool, fp_rates):
    result = []
    for threshold in sorted(yhat)[::-1]:
        pred_bool = yhat >= threshold
        fp, sensitivity = calc_fp_sensitivity(pred_bool, target_bool)
        fp_per_scan = fp / N_SCANS
        result.append([fp, fp_per_scan, sensitivity])

        if fp_per_scan > fp_rates[-1]:
            break
    
    result = pd.DataFrame(result)
    result.columns = ['fp', 'fp_per_scan', 'sensitivity']
    
    froc = []

    for fp_rate in fp_rates:
        i = result[(result['fp_per_scan'] <= fp_rate)].index.max()
        sensitivity = result.iloc[i]['sensitivity']
        froc.append(sensitivity)
    
    return froc


N_SCANS = 176
FP_RATES = [1/8, 1/4, 1/2, 1, 2, 4, 8]


df = pd.read_csv('results.csv')
df['baseline'] = torch.sigmoid(torch.Tensor(df['baseline'])).numpy()
df['transfer'] = torch.sigmoid(torch.Tensor(df['transfer'])).numpy()
df['y'] = (df['y'] == 1)


baseline_froc = calc_froc(df['baseline'], df['y'], FP_RATES)
transfer_froc = calc_froc(df['transfer'], df['y'], FP_RATES)

print(baseline_froc, np.mean(baseline_froc))
print(transfer_froc, np.mean(transfer_froc))
