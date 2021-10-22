# DeepForest-pytorch

* training data from "/orange/idtrees-collab/annotations"
* annotations are in file "/orange/idtrees-collab/annotations/hand_annotations.csv"
* split_big_images.py used to create train, test, and validation datasets from files in annotations

1. use preprocess2.py to convert benchmark dataset stored in NeonEvaluation into fully annotated dataset
2. use make_niwo.py or make_teak.py to make region specific datasets from the data created in step 1.

1. use preprocess3.py to create a dataset with CHM data.  dataset created from NeonEvaluation dataset.  4th channel is CHM
2. 
