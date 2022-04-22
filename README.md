# SCOP classification using Transformers

#### Data
* SCOP data: [URL](https://scop.mrc-lmb.cam.ac.uk/files/scop-cla-latest.txt)

#### Model hyperparameters


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
* Copy: `data/splits/cleaned_after_feature_computation.txt` to `data/splits/all_cleaned.txt`
* Check feature correctness: `python generators/check_feature_correctness.py`
    * Input file: `data/splits/all_cleaned.txt`
* Divide the cleaned data into train/val/test set as 70/15/15: `python generators/train_val_test_split.py`
    * Input file: `data/splits/all_cleaned.txt`
    * Output file:
        * `data/splits/train_{len(train)}.txt`
        * `data/splits/val_{len(val)}.txt`
        * `data/splits/test_{len(test)}.txt`
* Exclude classes that has less than n (default 10) data points.
    * Input file: `data/splits/all_cleaned.txt`
    * Output file: 
        * `data/splits/excluding_classes_having_less_than_n_datam.txt`
        * `data/splits/only_excluded_classes_having_less_than_n_datam.txt`
* Copy: `data/splits/excluding_classes_having_less_than_n_datam.txt` to `data/splits/all_cleaned_excluded.txt`
* Divide the excluded data into train/val/test set as 70/15/15: `python generators/train_val_test_split.py`
    * Input file: `data/splits/all_cleaned_excluded.txt`
    * Output file:
        * `data/splits/train_{len(train)}.txt`
        * `data/splits/val_{len(val)}.txt`
        * `data/splits/test_{len(test)}.txt`
* Analyze data to setup hyperparameters: `python analyzers/data.py`
* Train and test the model: `python models/train_test.py`

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