import sys
sys.path.append("../scop_classification")

import numpy as np
import utils as Utils
import seaborn as sn
import matplotlib.pyplot as plt

# scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification/outputs/images/* outputs/images/
val_result = Utils.load_pickle("outputs/images/val_cm.pkl")
print(val_result["loss"])

# val_cm = np.random.randint(100, size=(10, 10))
# np.savetxt("outputs/images/val_cm.csv", val_cm, delimiter=",")

# print(val_cm.shape)
# sn.heatmap(val_cm[:20, :20], annot=True)
# plt.show()

# test_cm = Utils.load_pickle("outputs/images/test_cm.pkl")