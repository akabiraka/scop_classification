import sys
sys.path.append("../scop_classification")

from features.AAWaveEncoding import AAWaveEncoding
import numpy as np
import matplotlib.pyplot as plt

Enc = AAWaveEncoding()
plt.figure(figsize=(10, 5))
n, dim = 20, 20
for i in range(n):
    y = np.array([Enc.get(i, dim) for i in range(n)])
    plt.plot(np.arange(n), y[:, 4:8])
    plt.scatter(np.repeat(i, dim), y[i])
    plt.legend(["dim %d"%p for p in range(4, 8)])
plt.show()
# plt.savefig(f"outputs/images/amino_acid_wave_encoding.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)