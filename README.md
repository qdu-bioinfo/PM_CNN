# PM_CNN
## Contents
* Introduction
* Package requirement
* Installation
* Data preprocess
* Model training and prediction
* Contact
## Introduction
In this study, we proposed a new deep learning framework PM-CNN (Phylogenetic Multi-path Convolutional Neural Network), which combines convolutional neural networks and microbial phylogenetic structures to predict various human diseases.
## Package requirement
* torch >= 1.11.0
* R >= 4.2.1
* numpy >= 1.22.3
* scipy >= 1.8.1
* scikit-learn >= 1.1.1
* matplotlib >= 3.5.1

Download PM-CNN:
```
git clone https://github.com/qdu-bioinfo/PM_CNN
```
## Installation
Install requirements:

```
pip install -r requirements.txt
```
## Data preprocess

First of all, we need to preprocess the data to get the information we want from the original data. Here, we take the gut data set as an example to demonstrate. In addition, after downloading our project files and configuring the relevant environment, if you want to run PM-CNN quickly, you can jump directly to the Model training and prediction section, copy the code and run it.

table1:
| OTU_id | Count | Abundance |
| ------ | ----- | --------- |
| 0      | 5     | 0.03       | 
| 1      | 10    | 0.1       |


table2:
| OTU_id | Count | Abundance |
| ------ | ----- | --------- |
| 0      | 4     | 0.02      |
| 1      | 2     | 0.001     |

Merge all tables:

```
python preprocess/preprocess.py
```

Merged abundance table:

| OTU1  | OTU2 | OTU3   | OTU4  | label |
| ----- | ---- | ------ | ----- | ----- |
| 0.03  | 0    | 0.001  | 0.001 | 1     |
| 0     | 0.01 | 0.1    | 0     | 2     |
| 0.002 | 0    | 0.004 | 0     | 3     |
| 0      | 0.02     |  0      |  0.003     |   2    |


Remove features with a 0 value ratio greater than 90%:

```
python proprecess/delete_feature.py
```
Get representative sequence:

```
python proprecess/get_represent_seqs.py
```
The evolutionary tree of the representative sequence is constructed by FastTree and Mafft, and the distance matrix is obtained by using the cophenetic function in the R package ape. The related software can be downloaded to the official website, or please contact my e-mail. The commands involved are as follows:

Mafft(Multiple sequence alignment):

```
mafft --auto input.fasta > output.fasta
```
FastTree(Construct a phylogenetic tree):

```
FastTree -nt -gtr output.fasta > gut.tree
```
R script(Get distance matrix):

```
Rscript proprecess/get_dis_matrix.R
```
After obtaining the distance matrix, we aim to obtain the phylogenetic correlation between OTU. Therefore, the next step is to transform the distance matrix into the correlation matrix by the designed distance transformation formula, and then carry out hierarchical clustering based on the correlation matrix, and finally get the result of multi-layer clustering.

Distance transformation and hierarchical clustering:

```
python model/clustering.py
```

Example diagram of clustering results:

![](https://markdown.liuchengtu.com/work/uploads/upload_96c134c0081ccd7afdc99e52cc4b49b5.jpg)


Get the result of multi-layer clustering:

```
python model/get_features.py
```

## Model training and prediction
Now, we have the training data and test data needed by the model. Next, we will begin to train the model and test the model, and before that, we need to introduce the usage of the model.

### Usage
You can see the parameters of PM-CNN through the comment information:

```
usage: model/PM_cnn.py[--train_x] [--train_y] //training set directory
                                        [--test_x] [--test_y] //test set directory.
                                        [--sequence] //clustering results
                                        [--batch_size] //batch_size
                                        [--learning_rate] //learning rate
                                        [--channel] //number of input and output channels
                                        [--kernel_size] //Convolution kernel size
                                        [--strides] //strides size
```

### Example

The human gut microbiome data contains 3113 samples with 5597 OTUs. See our paper for description details. Also, in order for you to run our program quickly, we integrated the training and testing parts of the model. The output results can be viewed in the console, or please move to the result folder, we have saved 10 running results in advance for your quick viewing.

```
cd PM_CNN/model
```

```
python PM_cnn.py --train_x ../data/Gut/train_data/X_train_5597.csv --train_y ../data/Gut/train_data/y_train_5597.csv --test_x ../data/Gut/test_data/X_test_5597.csv --test_y ../data/Gut/test_data/y_test_5597.csv --sequence ../data/Gut/Gut_feature.csv --batch_size 64 --epoch 34 --learning_rate 5e-3 --channel 16 --kernel_size 8 --strides 4
```

## Contact
All problems please contact PM-CNN development team:**Xiaoquan Su**    Email:[suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
