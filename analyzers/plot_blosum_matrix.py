import sys
from tkinter import font
sys.path.append("../scop_classification")
import features.static as STATIC
import blosum as bl
import numpy as np
import matplotlib.pyplot as plt

blosum_x = bl.BLOSUM(62)
blosum_matrix = np.zeros(shape=(20, 20))


for (key, value) in dict(blosum_x).items():
    # print(key, value)
    # key[0], key[1]
    i, j = STATIC.AA_1_LETTER.find(key[0]), STATIC.AA_1_LETTER.find(key[1])
    if i==-1 or j==-1: continue
    blosum_matrix[i, j] = value

np.fill_diagonal(blosum_matrix, 0)
# blosum_matrix = np.where(blosum_matrix<-.5, -4, blosum_matrix)
# print(blosum_matrix)    
plt.imshow(blosum_matrix, interpolation='none', cmap="YlGn")
plt.xticks(range(20), list(STATIC.AA_1_LETTER))
plt.yticks(range(20), list(STATIC.AA_1_LETTER))
plt.colorbar()

# Loop over data dimensions and create text annotations.
for i in range(len(STATIC.AA_1_LETTER)):
    for j in range(len(STATIC.AA_1_LETTER)):
        text = plt.text(j, i, int(blosum_matrix[i, j]), ha="center", va="center", color="b", fontsize="xx-small")
# plt.show()
plt.savefig(f"outputs/images/blosum_matrix.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.05)
