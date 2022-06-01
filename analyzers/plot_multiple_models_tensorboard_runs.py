import sys
sys.path.append("../scop_classification")

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_runs(paths, model_names, x_label, y_label, tag, img_name, img_format="png"):
    plt.cla()
    for i, path in enumerate(paths):
        ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
        _absorb_print = ea.Reload()

        x, y=[],[]
        for j, event in enumerate(ea.Scalars(tag)):
            # print(event.step, event.value)
            x.append(event.step)
            y.append(event.value) 
            if j==700: break
        if "loss" in tag: plt.plot(x, y, label=model_names[i]+f" (best={np.min(y):.3f})")
        elif "acc" in tag: plt.plot(x, y, label=model_names[i]+f" (best={np.max(y):.3f})")
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig(f"outputs/images/{img_name}.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)

dir="outputs/tensorboard_runs/"

# configs for plotting best model performance using contact-map
# paths = [dir+"CW_1e-05_64_300_cuda/events.out.tfevents.1652220562.node056.orc.gmu.edu.8499.0",
#         dir+"NoAttnMask_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652452872.node056.orc.gmu.edu.29944.0",
#         dir+"Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652397952.dgx002.orc.gmu.edu.1563475.0",
#         dir+"Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652371630.NODE050.orc.gmu.edu.30609.0"]
# model_names = ["FT-PRoBERTa", "ProToFormer (128-SEQ)", "ProToFormer (128-SEQ+CM)", "ProToFormer (256-SEQ+CM)"]
# img_names = ["comparison_train_loss", "comparison_val_loss", "comparison_val_acc"]


#configs for ablation study on topological maps
paths = [dir+"Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652371630.NODE050.orc.gmu.edu.30609.0",
         dir+"Model_distmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652807314.node056.orc.gmu.edu.27957.0",
         dir+"Model_noattnmask_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/events.out.tfevents.1653673213.NODE050.orc.gmu.edu.27628.0",
         dir+"Model_longrange_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/events.out.tfevents.1653677477.node056.orc.gmu.edu.7423.0",
         dir+"Model_nobackbone_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/events.out.tfevents.1653676883.node056.orc.gmu.edu.5229.0"] 
model_names = ["ProToFormer (256-SEQ+CM)", "ProToFormer (256-SEQ+IDM)", "ProToFormer (256-SEQ+NAM)", "ProToFormer (256-SEQ+LR)", "ProToFormer (256-SEQ+NBB)"]        
img_names = ["topomaps_usage_comparison_train_loss", "topomaps_usage_comparison_val_loss", "topomaps_usage_comparison_val_acc"]

# plotting using above set of configurations
tags = ["train loss", "val loss", "acc"]
y_labels = ["Cross-entropy", "Cross-entropy", "Accuracy"]
x_label = "Number of epochs"

for i, tag in enumerate(tags):
    plot_tensorboard_runs(paths, model_names, x_label, y_label=y_labels[i], tag=tags[i], img_name=img_names[i], img_format="png")
