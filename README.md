# SCOP classification using Transformers

#### Data
* SCOP data: [URL](https://scop.mrc-lmb.cam.ac.uk/files/scop-cla-latest.txt)

#### Model hyperparameters


#### Workflow
* Separate class labels from the downloaded dataset: `python generators/data_clean.py`
* Download PDB, clean and generate fasta: `python generators/DownloadCleanFasta.py`
* Clean before/after download data: `python generators/data_helper.py`
* Generate features: `python generators/Features.py`
* Divide the data into train/val/test set as 70/15/15: `python generators/train_val_test_split.py`
* Analyze data to setup hyperparameters: `python analyzers/data.py`
* Train and test the model: `python models/train_test.py`

#### Data issues and resolution
* If len(chain_id)>1: removed
* 3iyo:D contains CA for all residues: removed
* If DSSP cannot compute SS/RASA for a residue_id, the value is set as SS="-" (Coil) and RASA=0.5 (neither exposed nor buried).
* If an atom (i.e CA or CB) is not found while computing Contact-map, the value is set as 0.0 as they are in contact.
* If a chain region looks like this "A:1-272,A:312-378", skip.