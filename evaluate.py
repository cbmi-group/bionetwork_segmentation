import os
import glob
import numpy as np
import cv2
import time
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance_val(prd, label):

    n_imgs = np.shape(prd)[0]
    total_max = 0
    total_min = 0

    for i in range(n_imgs):
        non_zero_prd = np.transpose(np.nonzero(prd[i]))
        non_zero_mask = np.transpose(np.nonzero(label[i]))
        h_dist_max = max(directed_hausdorff(non_zero_prd, non_zero_mask)[0], directed_hausdorff(non_zero_mask, non_zero_prd)[0])
        h_dist_min = min(directed_hausdorff(non_zero_prd, non_zero_mask)[0], directed_hausdorff(non_zero_mask, non_zero_prd)[0])
        total_max += h_dist_max
        total_min += h_dist_min

    mean_max = total_max / n_imgs
    mean_min = total_min / n_imgs

    return mean_max, mean_min

if __name__ == "__main__":

    order = False

    test_data_dir = "./datasets/test_er.txt"
    score_map_dir = './train_log/er_train_aug_v1_20200727_bceloss/predict_score/test_er_30'
    with open(test_data_dir, "r") as fid:
        lines = fid.readlines()

    masks_list = []
    for line in lines:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split(" ")
        masks_list.append(words[1])

    if order == True:
        masks_list = sorted(masks_list, key=lambda s: int(os.path.split(s)[1][0:-4]))

    img_num = len(masks_list)

    y_true = []
    y_scores = []

    hausdorff_prd = []
    hausdorff_lable = []

    for mask_path in masks_list:
        mask_name = os.path.splitext(os.path.split(mask_path)[1])[0]
        score_path = os.path.join(score_map_dir, mask_name+".npy")
        score = np.load(score_path)

        print("==> Read score map: %s." % (score_path))
        label = cv2.imread(mask_path, -1) / 255
        label = label.astype(np.uint8)
        y_true.append(label.flatten())
        y_scores.append(score.flatten())

        hausdorff_prd.append(score)
        hausdorff_lable.append(label)


    thresholds = np.arange(0.1, 0.9, 0.05)[1:]
    y_true, y_scores = np.concatenate(y_true, axis=0), np.concatenate(y_scores, axis=0)
    hausdorff_scores = np.array(hausdorff_prd)
    hausdorff_true = np.array(hausdorff_lable)

    acc = np.zeros(len(thresholds))
    specificity = np.zeros(len(thresholds))
    sensitivity = np.zeros(len(thresholds))
    precision = np.zeros(len(thresholds))
    iou = np.zeros(len(thresholds))
    f1 = np.zeros(len(thresholds))
    hausdorff_max = np.zeros(len(thresholds))
    hausdorff_min = np.zeros(len(thresholds))

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.uint8)
        hausdorff_seg = (hausdorff_scores > threshold).astype(np.uint8)
        confusion = confusion_matrix(y_true, y_pred)
        tp = float(confusion[1, 1])
        fn = float(confusion[1, 0])
        fp = float(confusion[0, 1])
        tn = float(confusion[0, 0])

        acc[indy] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[indy] = tp / (tp + fn)
        specificity[indy] = tn / (tn + fp)
        precision[indy] = tp / (tp + fp)
        f1[indy] = 2 * sensitivity[indy] * precision[indy] / (sensitivity[indy] + precision[indy])
        sum_area = (y_pred + y_true)
        union = np.sum(sum_area == 1)
        iou[indy] = tp / float(union + tp)

        hausdorff_max[indy], hausdorff_min[indy] = hausdorff_distance_val(hausdorff_seg, hausdorff_true)


        print('threshold {:.10f} ==>hdf: {:.4f}, iou: {:.4f}, f1 score: {:.4f}, acc: {:.4f}, sen: {:.4f}, spec: {:.4f}'.format(threshold, hausdorff_max[indy], iou[indy], f1[indy], acc[indy], sensitivity[indy], specificity[indy]))

    thred_indx = np.argmax(iou)
    m_hausdorff_max = hausdorff_max[thred_indx]
    m_hausdorff_min = hausdorff_min[thred_indx]
    m_iou = iou[thred_indx]
    m_f1 = f1[thred_indx]
    m_acc = acc[thred_indx]
    m_spc = specificity[thred_indx]
    m_sec = sensitivity[thred_indx]
    m_auc = roc_auc_score(y_true, y_scores)
    print("==> Threshold: %.9f." % (thresholds[thred_indx]))
    print("==> Hau_min: %.2f." % (m_hausdorff_min))
    print("==> Hau_max: %.2f." % (m_hausdorff_max))
    print("==> IoU: %.4f." % (m_iou))
    print("==> F1: %.4f." % (m_f1))
    print("==> AUC: %.4f." % (m_auc))
    print("==> Spc: %.4f." % (m_spc))
    print("==> Sec: %.4f." % (m_sec))
    print("==> ACC: %.4f." % (m_acc))
    print(score_map_dir)
    print(img_num)



