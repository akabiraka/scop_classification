import sys
sys.path.append("../scop_classification")
import utils as Utils
import features.static as STATIC
import blosum as bl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# download aa-embeddings using the following command by changing the model name
# scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification/outputs/predictions/Model_nobackbone_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/20_aa_embeddings.pkl outputs/predictions/Model_nobackbone_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/

aa_20_embeddings = Utils.load_pickle("outputs/predictions/Model_nobackbone_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/20_aa_embeddings.pkl")
# aa_20_embeddings = aa_20_embeddings.squeeze(0).cpu().detach().numpy()[:20, :] # temp line, remove this, this transformation is added while saving
print(aa_20_embeddings.shape)
relation_matrix = cosine_similarity(aa_20_embeddings)
print(relation_matrix.shape)
np.fill_diagonal(relation_matrix, 0)
# relation_matrix = np.where(relation_matrix<0, -.1, relation_matrix)

plt.imshow(relation_matrix, interpolation='none', cmap="YlGn")
plt.xticks(range(20), list(STATIC.AA_1_LETTER))
plt.yticks(range(20), list(STATIC.AA_1_LETTER))
plt.colorbar()
for i in range(len(STATIC.AA_1_LETTER)):
    for j in range(len(STATIC.AA_1_LETTER)):
        text = plt.text(j, i, f"{relation_matrix[i, j]:.1f}", ha="center", va="center", color="b", fontsize="xx-small")
# plt.show()
plt.savefig(f"outputs/images/relation_of_20_aa_embeddings.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.05)

# blosum_x = bl.BLOSUM(62)
# blosum_matrix = np.zeros(shape=(20, 20))


# for (key, value) in dict(blosum_x).items():
#     # print(key, value)
#     # key[0], key[1]
#     i, j = STATIC.AA_1_LETTER.find(key[0]), STATIC.AA_1_LETTER.find(key[1])
#     if i==-1 or j==-1: continue
#     blosum_matrix[i, j] = value

# pearson = np.corrcoef(relation_matrix,blosum_matrix)

# plt.imshow(pearson, interpolation='none', cmap="YlGn")
# plt.xticks(range(20), list(STATIC.AA_1_LETTER))
# plt.yticks(range(20), list(STATIC.AA_1_LETTER))
# plt.colorbar()
# plt.show()



# fig, ax = plt.subplots()
# im = ax.imshow(blosum_matrix, cmap="YlGn")

# # Show all ticks and label them with the respective list entries
# ax.xticks(np.arange(len(STATIC.AA_1_LETTER)), labels=list(STATIC.AA_1_LETTER))
# # ax.set_yticks(np.arange(len(STATIC.AA_1_LETTER)), labels=list(STATIC.AA_1_LETTER))

# # Loop over data dimensions and create text annotations.
# # for i in range(len(STATIC.AA_1_LETTER)):
# #     for j in range(len(STATIC.AA_1_LETTER)):
# #         text = ax.text(j, i, blosum_matrix[i, j],
# #                        ha="center", va="center", color="w")

# fig.tight_layout()
# plt.show()