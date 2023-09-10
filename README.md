# DSB2-SKUnet
大學時期的大專生計畫(已結案)

### Overview
In the 3D cardiac image segmentation task, we utilize three public datasets from Kaggle for model training and evaluate on the testing dataset from Kaggle. The model architecture is U-Net combined with two attention modules, CBAM and SK-Net. To fit the ground truth format in Kaggle (i.e., cardiac output, ml), we transfer the segmentation map (cm2) to the cardiac output (ml) based on the resize ratio in the preprocessing step and the patient's information from the DICOM file.

### Note
merge_normal_data.py  >>  merge 3 datasets (SunnyBrook data and some manual data) and split into 3 folders

### Environment
Python : 3.7.7

tf :2.1
