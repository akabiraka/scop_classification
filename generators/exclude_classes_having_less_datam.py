import sys
sys.path.append("../scop_classification")
import pandas as pd

def exclude_classes_having_less_than_n_datams(inp_file, out_file1, out_file2, cls_col_name="SF", n=10):
    df = pd.read_csv(inp_file)
    df['counts'] = df[cls_col_name].map(df[cls_col_name].value_counts())
    exclude_mask = df['counts']>n
    # print(exclude_mask)

    df_excluding_classes = df[exclude_mask]
    df_excluding_classes.reset_index(drop=True, inplace=True)
    print(df_excluding_classes.shape)
    # print(df_excluding_classes)
    df_excluding_classes.to_csv(out_file1, index=False)

    df_only_excluded_classes = df[~exclude_mask]
    df_only_excluded_classes.reset_index(drop=True, inplace=True)
    print(df_only_excluded_classes.shape)
    # print(df_only_excluded_classes)
    df_only_excluded_classes.to_csv(out_file2, index=False)

    
exclude_classes_having_less_than_n_datams("data/splits/all_cleaned.txt", 
            "data/splits/excluding_classes_having_less_than_n_datam.txt",
            "data/splits/only_excluded_classes_having_less_than_n_datam.txt", 
            cls_col_name="SF", n=10)