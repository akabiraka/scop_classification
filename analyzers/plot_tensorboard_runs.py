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

paths = [dir+"CW_1e-05_64_300_cuda/events.out.tfevents.1652220562.node056.orc.gmu.edu.8499.0",
        dir+"NoAttnMask_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652452872.node056.orc.gmu.edu.29944.0",
        dir+"Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652397952.dgx002.orc.gmu.edu.1563475.0",
        dir+"Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652371630.NODE050.orc.gmu.edu.30609.0"]

model_names = ["FT-BERT", "ProToFormer (seq+128)", "ProToFormer (seq+128+CM)", "ProToFormer (seq+256+CM)"]

tags = ["train loss", "val loss", "acc"]
y_labels = ["Cross-entropy", "Cross-entropy", "Accuracy"]
img_names = ["comparison_train_loss", "comparison_val_loss", "comparison_val_acc"]
x_label = "Number of epochs"

for i, tag in enumerate(tags):
    plot_tensorboard_runs(paths, model_names, x_label, y_label=y_labels[i], tag=tags[i], img_name=img_names[i], img_format="png")