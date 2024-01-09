# PM_CNN

## Contents

* Introduction

* Package requirement

* Installation

* Program running process:
  
  * Step 1: Data preprocess(optional)
  
  * Step 2: Build a phylogenetic tree(optional)
  
  * Step 3: Get correlation matrix
  
  * Step 4: Distance transformation and hierarchical clustering
  
  * Step 5: Model training and testing

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

### Download PM-CNN:

```
git clone https://github.com/qdu-bioinfo/PM_CNN
```

## Installation

### Install requirements:

```
pip install -r requirements.txt
```

## Program running process:

### Step 1: Data preprocess

In this step, the sample abundance table needs to be obtained based on the abundance information of each sample and related meta information. We provide additional programs to support this input type. If you want to run PM-CNN on any other data set, you only need to modify the relevant path information according to the usage requirements below. If you already have the OTU abundance table of all samples, you can skip this step. If you want to run PM-CNN quickly, you can jump directly to the "Model training and testing" section and run the relevant commands.

#### Input:

#### sample1:                                  
| OTU_id | Count | Abundance |
| ------ | ----- | --------- |				
| 0      | 5     | 0.03      | 				
| 1      | 10    | 0.1       |

#### sample2:
| OTU_id | Count | Abundance |				
| ------ | ----- | --------- |				
| 0      | 4     | 0.02      |
| 1      | 2     | 0.001     |

#### meta information:
| SampleID | Status |
| ------ | ----- |
| sample1  |    Healthy    |
| sample2  |    Healthy    |
| sample3  |  Gingivitis   |
| sample4  | Periodontitis |


#### Usage:

```
preprocess/preprocess.py [--input] [-i] //Input the storage path of all samples
		         [--meta] [-m] //Meta information of all samples
                         [--output] [-o] //Output the merged sample abundance table
```

#### Example running:

```
python preprocess.py --input ../data/Gut/Raw_data --meta ../data/Gut/Gut_3113_meta.csv --output ../data/Gut_Abundance_table.xlsx
```

### Step 2: Build a phylogenetic tree(optional)

This step is optional, the phylogeny tree can either be constructed from representative sequences (e.g. marker gene or amplicon), or provided by users from a NEWICK format file (e.g. for shotgun). The evolutionary tree of the representative sequence is constructed by FastTree and Mafft. The related software can be downloaded to the official website, or please contact my e-mail. The commands involved are as follows:

Get representative sequence:

```
usage: example/code/get_represent_seqs.py [--input] [-i] //Input CSV file path
                                          [--output] [-o] //Output TXT file path
```

#### Example running:

```
python get_represent_seqs.py --input ../data/del_ex_Abundance_table.csv --output ../data/ex_represent.txt  // you need to download GreenGenes database
```

Mafft(Multiple sequence alignment):

```
mafft --auto example/data/ex_respresent.txt > example/data/output.fasta // you need to download Mafft
```

FastTree(Construct a phylogenetic tree):

```
FastTree -nt -gtr example/data/output.fasta > example/data/ex.tree  // you need to download FastTree
```

### Step 3: Get correlation matrix

The Cophenetice distance matrix is obtained by using the cophenetic function in the R package ape. The related software can be downloaded to the official website, or please contact my e-mail.

R script(Get cophenetic distance matrix):

```
usage: example/code/get_represent_seqs.py [--input] [-i] //Input tree file
                                          [--output] [-o] //Output distance matrix file
```

#### Example running:

```
Rscript get_dis_matrix.R --input example/data/ex.tree --output example/data/ex_distance_matrix.csv
```

After obtaining the distance matrix, we aim to obtain the phylogenetic correlation between OTU. Therefore, the next step is to transform the distance matrix into the correlation matrix by the designed distance transformation formula, and then carry out hierarchical clustering based on the correlation matrix, and finally get the result of multi-layer clustering.

### Step 4: Distance transformation and hierarchical clustering

```
usage: example/code/get_represent_seqs.py [--input] [-i] //Input original distance matrix csv file
                                          [--tran] [-o] //Get distance converted csv file
                                          [--npy] [-n] //Convert the converted matrix csv to npy format
                                          [--output] [-o] //Hierarchical clustering result path
```

#### Example running:

```
python model/clustering.py --input ../data/Oral/oral_1554.csv --tran ../data/Oral/oral_1554_tran.csv --npy ../data/Oral/oral_1554_tran.npy --output ../data/Oral/Oral_feature.csv
```

### Step 5: Model training and prediction

Now, we have the training data and test data needed by the model. Next, we will begin to train the model and test the model, and before that, we need to introduce the usage of the model.

### Usage

#### You can see the parameters of PM-CNN through the comment information:

```
usage: model/PMCNN.py [--train] //train PM-CNN model [--test] //test PM-CNN model
                                        [--train_x] [--train_y] //training set directory
                                        [--test_x] [--test_y] //testing set directory.
                                        [--sequence] //clustering results
                                        [--res] //ROC curve for each label
                                        [--batch_size] //batch_size
                                        [--batch_norm] //batch normalization
                                        [--test_num] //number of test sets
                                        [--label_sum] //total number of labels
                                        [--feature_sum] //total number of features
                                        [--shape] //total number of neurons after channel merging
                                        [--learning_rate] //learning rate
                                        [--channel] //number of input and output channels
                                        [--kernel_size] //Convolution kernel size
                                        [--strides] //strides size
```

#### Example running(Dataset 1):

The human oral microbiome data contains 1587 samples with 1554 OTUs. Contains three label types: Control, Periodontitis, and Gingivitis, which are marked with 0, 1, and 2 respectively. Then, divide 70% of the samples into the training set and 30% of the samples into the test set. See our paper for description details. Also, in order for you to run our program quickly, we integrated the training and testing parts of the model. The output results can be viewed in the console, or please move to the result folder, we have saved 10 running results in advance for your quick viewing.

```
cd PM_CNN/model
```

#### Training PM-CNN:

```
python PMCNN.py --train --train_x ../data/Oral/train_data/X_train_1554.csv --train_y ../data/Oral/train_data/y_train_1554.csv --sequence ../data/Oral/Oral_feature.csv --save_model ../data/Oral/PM-CNN_model.pth --batch_norm 64 --label_sum 3 --feature_sum 1554 --shape 24576 --batch_size 32 --epoch 35 --learning_rate 5e-3 --channel 64 --kernel_size 8 --strides 4 --res ../result/
```

#### Testing PM-CNN:

```
python PMCNN.py --test --test_x ../data/Oral/test_data/X_test_1554.csv --test_y ../data/Oral/test_data/y_test_1554.csv --test_num 477 --label_sum 3 --res ../result/
```

#### Example running(Dataset 2):

The human gut microbiome data contains 3113 samples with 5597 OTUs. Contains five label types: Control, IBD, HIV, EDD, and CRC, which are marked with 0, 1, 2, 3, and 4 respectively. Then, divide 70% of the samples into the training set and 30% of the samples into the test set. See our paper for description details. The output results can be viewed in the console, or please move to the result folder, we have saved 10 running results in advance for your quick viewing.

#### Training PM-CNN:

```
python PMCNN.py --train --train_x ../data/Gut/train_data/X_train_5597.csv --train_y ../data/Gut/train_data/y_train_5597.csv --sequence ../data/Gut/Gut_feature.csv --save_model ../data/Gut/PM-CNN_model.pth --batch_norm 64 --label_sum 5 --shape 22336 --feature_sum 5597 --batch_size 64 --epoch 45 --learning_rate 5e-3 --channel 16 --kernel_size 8 --strides 4
```

#### Testing PM-CNN:

```
python PMCNN.py --test --test_x ../data/Gut/test_data/X_test_5597.csv --test_y ../data/Gut/test_data/y_test_5597.csv --test_num 934 --save_model ../data/Gut/PM-CNN_model.pth --res ../result/
```

## Contact

#### All problems please contact PM-CNN development team:**Xiaoquan Su**    Email:[suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
