# ProToFormer: Sequence-Structure Embeddings via Protein Language Models Improve on Prediction Tasks

#### Data
* SCOP data: [URL](https://scop.mrc-lmb.cam.ac.uk/files/scop-cla-latest.txt)

#### Model hyperparameters
    * task="SF"
    * max_len=1024           # maximum length of which the dataset generation process will padd/truncate
    * dim_embed=20           # dim_embed must be divisible by num_head
    * n_attn_heads=10        # number of attention heads
    * dim_ff=2*dim_embed     # feed forward network dimension is set as 2*dim_embed
    * n_encoder_layers=6
    * dropout=0.3
    * init_lr=0.001
    * n_epochs=300
    * batch_size=50


#### Workflow
* Separate class labels from the downloaded dataset: `python generators/separate_class_labels.py`
    * Input file: `data/splits/scop-cla-latest.txt`
    * Output file: `data/splits/cleaned_after_separating_class_labels.txt`
* Download PDB, clean and generate fasta: `python generators/DownloadCleanFasta.py`
    * Input file: `data/splits/cleaned_after_separating_class_labels.txt`
    * Output file: `data/splits/cleaned_after_pdbs_downloaded.txt`
* Update sequence length: `python generators/update_seq_length.py`
    * Input/Output file: `data/splits/cleaned_after_pdbs_downloaded.txt`
* Generate features: `python generators/Features.py`
    * Input file: `data/splits/cleaned_after_pdbs_downloaded.txt`
    * Output file: `data/splits/cleaned_after_feature_computation.txt`
* Copy: `cp data/splits/cleaned_after_feature_computation.txt data/splits/all_cleaned.txt`
* Check feature correctness: `python generators/check_feature_correctness.py`
    * Input file: `data/splits/all_cleaned.txt`
    * The issues will be reported in: `data/splits/tracebacks.txt`
* Divide the cleaned data into train/val/test set as 70/15/15: `python generators/train_val_test_split.py`
    * Input file: `data/splits/all_cleaned.txt`
    * Output file:
        * `data/splits/train_24538.txt`: (24538, 17)
        * `data/splits/val_4458.txt`: (4458, 17)
        * `data/splits/test_5862.txt`: (5862, 17)
* Exclude classes that has less than n (default 10) data points.
    * Input file: `data/splits/all_cleaned.txt`
    * Output file: 
        * `data/splits/excluding_classes_having_less_than_n_datam.txt`
        * `data/splits/only_excluded_classes_having_less_than_n_datam.txt`
* Copy: `cp data/splits/excluding_classes_having_less_than_n_datam.txt data/splits/all_cleaned_excluded.txt`
* Divide the excluded data into train/val/test set as 70/15/15: `python generators/train_val_test_split.py`
    * Input file: `data/splits/all_cleaned_excluded.txt`
    * Output file:
        * `data/splits/train_19828.txt`: (19828, 18)  
        * `data/splits/val_4106.txt`: (4106, 18)
        * `data/splits/test_4410.txt`: (4410, 18)
* Analyze data to setup hyperparameters: `python analyzers/data.py`
* Train and test the model: `python models/train_test.py`

#### Analyze
* To analize data: `python analyzers/data.py`
* To analize single datam when generating dataset: `python analyzers/dataset.py`
* To vizualize the training progress: `tensorboard --logdir=outputs/tensorboard_runs/`
* To download the runs outputs: `scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification/outputs/tensorboard_runs/* outputs/tensorboard_runs/`
* To download the val/test outputs: `scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification/outputs/predictions/* outputs/predictions/`

#### Data issues and resolution
* If len(chain_id)>1: removed
* 3iyo:D contains CA for all residues: removed
* If DSSP cannot compute SS/RASA for a residue_id, the value is set as SS="-" (Coil) and RASA=0.5 (neither exposed nor buried).
* If an atom (i.e CA or CB) is not found while computing Contact-map, the value is set as 0.0 as they are in contact.
* If a chain region looks like this "A:1-272,A:312-378", skip.
* 6qwj: does not exist
* 1ejg: Bio.PDB.PDBExceptions.PDBConstructionException: Blank altlocs in duplicate residue SER (' ', 22, ' ')
* 3msz: does not have 93 atom at chain A.
* DSSP failed (only contains CA atoms):

#### 3rd party Softwares
* DSSP installation: 
    * sudo apt install dssp, or
    * Install anaconda and then conda install -c salilab dssp
    * 
## Citation
If the model is found useful, we request to cite the relevant paper:
```bibtex
@INPROCEEDINGS{10030025,
  author={Kabir, Anowarul and Shehu, Amarda},
  booktitle={2022 IEEE International Conference on Knowledge Graph (ICKG)}, 
  title={Sequence-Structure Embeddings via Protein Language Models Improve on Prediction Tasks}, 
  year={2022},
  volume={},
  number={},
  pages={105-112},
  keywords={Location awareness;Soft sensors;Semantics;Training data;Predictive models;Transformers;Protein sequence;Protein language model;Transformer;Sequence structure transformer;Protein function;superfamily},
  doi={10.1109/ICKG55886.2022.00021}}
```
