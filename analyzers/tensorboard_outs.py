import sys
sys.path.append("../scop_classification")

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def parse_tensorboard(path, model_name=""):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # print(ea.Tags()["scalars"]) # shows the tags
    
    labels=["Train loss", "Val loss", "Val accuracy"]
    for j, tag in enumerate(["train loss", "val loss", "acc"]):
        x, y=[],[]
        for i, event in enumerate(ea.Scalars(tag)):
            # print(event.step, event.value)
            x.append(event.step)
            y.append(event.value)
            # plt.plot(event.step, event.value)
            # () 
            if i==175: break
        if "loss" in tag: plt.plot(x, y, label=model_name + " " +labels[j] + f" (best={np.min(y):.3f})")
        elif "acc" in tag: plt.plot(x, y, label=model_name + " " +labels[j] + f" (best={np.max(y):.3f})")
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross-entropy loss/Accuracy")
    

parse_tensorboard(path="outputs/tensorboard_runs/Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652397952.dgx002.orc.gmu.edu.1563475.0",
                  model_name="Our model")   

parse_tensorboard(path="outputs/tensorboard_runs/CW_1e-05_64_300_cuda/events.out.tfevents.1652220562.node056.orc.gmu.edu.8499.0",
                  model_name="PRoBERTa")                  

img_name="our_model_vs_proberta_performance_comparison"
img_format="png"
if img_name==None: plt.show()
else: plt.savefig(f"outputs/images/{img_name}.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)