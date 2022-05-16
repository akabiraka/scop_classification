import sys
sys.path.append("../scop_classification")
import pandas as pd

# returns mask of classes having at least th datams
def get_mask(df, cls_col_name, th):
    df['counts'] = df[cls_col_name].map(df[cls_col_name].value_counts())
    return  df['counts']>=th


# returns the class dictionary aliased
def get_class_dict(df, cls_col_name):
    x = df[cls_col_name].unique().tolist()
    class_dict = {j:i for i,j in enumerate(x)}
    n_classes = len(class_dict)
    print(f"n_classes: {n_classes}")
    # print(class_dict)
    return class_dict

# returns cls indices by masked out
def get_cls_idx(df, mask, cls_col_name, class_dict):
    df_excluding_classes = df[mask]
    df_excluding_classes.reset_index(drop=True, inplace=True)
    classes = df_excluding_classes[cls_col_name].unique()
    idx = [class_dict[x] for x in classes]
    return idx


# returns cls indices of having at lest th datams and less than th datas
def get_idx_set_and_idx_prime_set_of_th(inp_file_path="data/splits/all_cleaned.txt", cls_col_name="SF", th=10):
    df = pd.read_csv(inp_file_path)
    
    exclude_mask = get_mask(df, cls_col_name, th)
    class_dict = get_class_dict(df, cls_col_name)

    idx = get_cls_idx(df, exclude_mask, cls_col_name, class_dict) #idx_having_at_least_n_datams
    idx_prime = set(range(len(class_dict))) - set(idx) #idx_having_less_than_n_datams

    print(f"#cls indices having >={th} datams: {len(idx)}, #cls indices having <{th} datams: {len(idx_prime)}")

    return set(idx), idx_prime


# get_idx_set_and_idx_prime_set_of_th(inp_file_path="data/splits/all_cleaned.txt", cls_col_name="SF", th=10)

def get_refined_labels(true_labels, pred_labels, idx):
    new_true_labels, new_pred_labels = [], []
    for i, label in enumerate(true_labels):
        if label in idx:
            new_true_labels.append(label)
            new_pred_labels.append(pred_labels[i])
    return new_true_labels, new_pred_labels



