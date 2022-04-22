import sys
sys.path.append("../scop_classification")

import pandas as pd
import numpy as np


def random_split(inp_file_path, train_frac=0.7, val_frac=.15):
    """test = 1-train_frac+val_frac"""
    df = pd.read_csv(inp_file_path)
    print(len(df))
    np.random.seed(1)
    train, val, test = np.split(df.sample(frac=1, random_state=1), [int(train_frac*len(df)), int((train_frac+val_frac)*len(df))])
    return train, val, test


def split_by_having_all_classes(inp_file_path, cls_col_name="SF", train_frac=0.7, val_frac=.15):
    """todo
    train: select train_frac% for each class
    """
    df = pd.read_csv(inp_file_path)
    unique_classes = df[cls_col_name].unique().tolist()
    # print(len(unique_classes))
    
    train, df = sample_frac_for_each_class(df, unique_classes, cls_col_name, train_frac)
    # print(train.shape, df.shape)
    
    updated_val_frac = val_frac/(1-train_frac)
    val, df = sample_frac_for_each_class(df, unique_classes, cls_col_name, updated_val_frac)
    # print(val.shape, df.shape)
    
    test = df
    # print(test.shape, df.shape)
    return train, val, test

# helper function of split_by_having_all_classes
def sample_frac_for_each_class(df, classes, cls_col_name, frac):
    new_df = pd.DataFrame(columns=df.columns)
    for cls in classes:
        # sample rows
        cls_specific_df = df[df[cls_col_name]==cls] 
        sampled_df = cls_specific_df.sample(frac=frac)
        # drop sampled rows
        df = df.drop(sampled_df.index)
        df.reset_index(drop=True, inplace=True)
        # append sampled rows
        sampled_df.reset_index(drop=True, inplace=True)
        new_df = new_df.append(sampled_df, ignore_index=True)
        new_df.reset_index(drop=True, inplace=True)
        # print(new_df.shape, df.shape)
    return new_df, df

inp_file_path = "data/splits/cleaned_after_feature_computation.txt"
# train, val, test = random_split(inp_file_path, train_frac=0.7, test_frac=.3)
train, val, test = split_by_having_all_classes(inp_file_path, cls_col_name="SF", train_frac=0.7, val_frac=.15)
train.to_csv(f"data/splits/train_{len(train)}.txt", index=False)
val.to_csv(f"data/splits/val_{len(val)}.txt", index=False)
test.to_csv(f"data/splits/test_{len(test)}.txt", index=False)

print(train.shape, val.shape, test.shape)