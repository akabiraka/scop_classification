import sys
sys.path.append("../scop_classification")

import numpy as np
import utils as Utils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt

def print_acc_prec_rec_f1_score(target_classes, pred_classes):
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average="weighted")
    recall = recall_score(target_classes, pred_classes, average="weighted")
    f1 = f1_score(target_classes, pred_classes, average="weighted")
    print(f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")


def print_roc_auc_score(target_cls_distributions, pred_cls_distributions):
    roc_auc = roc_auc_score(target_cls_distributions, pred_cls_distributions, average="micro", multi_class="ovr")
    print(f"roc_auc: {roc_auc}")


def get_one_hot(labels):
    shape = (labels.size, labels.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1
    return one_hot

# scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification/outputs/predictions/* outputs/predictions/

# for fname in ["Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_val_result", "Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_test_result"]:
# for fname in ["NoAttnMask_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda_val_result", "NoAttnMask_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda_test_result"]:    
for fname in ["Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda_val_result", "Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda_test_result"]:        
    result = Utils.load_pickle(f"outputs/predictions/{fname}.pkl")

    loss = result["loss"]
    true_labels = np.array(np.stack(result["true_labels"]))[:, 0]
    pred_class_distributions = np.stack(result["pred_class_distributions"])

    target_class_distributions = get_one_hot(true_labels)
    pred_labels = np.argmax(pred_class_distributions, axis=1)

    print(loss, true_labels.shape, pred_class_distributions.shape, pred_labels.shape)

    print_acc_prec_rec_f1_score(true_labels, pred_labels)

    print_roc_auc_score(target_class_distributions, pred_class_distributions)

# cm = confusion_matrix(true_labels, pred_labels)
# print(cm.shape)

# print(classification_report(true_labels, pred_labels))
