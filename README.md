# PM_CNN

## Contents

* Introduction

* Package requirement

* Installation

* Program running process:
    
  * Step 1: Build a phylogenetic tree (optional)
  
  * Step 2: Get cophenetic distance matrix
  
  * Step 3: Distance transformation and hierarchical clustering
  
  * Step 4: Model training and testing
  
* Dataset introduction

* Contact

## Introduction

In this study, we proposed a new deep learning framework PM-CNN (Phylogenetic Multi-path Convolutional Neural Network), which combines convolutional neural networks and microbial phylogenetic structures to predict various human diseases.

### Package requirement:

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

## Installation:

```
pip install -r requirements.txt
```

## Program running process:

Prior to initiating the execution of PM-CNN, it is imperative to confirm the format of the input data. The input data should exclusively adhere to the microbial feature table structure, (e.g. OTU table or species table). Ensuring compliance with this prerequisite is crucial for the accurate and effective utilization of PM-CNN in the analysis. 

#### example:

#### OTU table:

|  | OTU1 | OTU2 | OTU3 | OTU4|
| ------ | ------ | ------ | ------ | ------ |
| sample1      | 0.01     | 0.03      | 0 | 0.06 |
| sample2      | 0.05    | 0       | 0.03 | 0.001 |
| sample3      | 0    | 0.2       | 0 | 0.01 |
| sample4      | 0    | 0.02       | 0.01 | 0.001 |

#### meta information:

| SampleID | Status        |
| -------- | ------------- |
| sample1  | Healthy       |
| sample2  | Healthy       |
| sample3  | Gingivitis    |
| sample4  | Periodontitis |



## Program running process

All the following processes are shown using the data in the example folder.

### Step 1: Build a phylogenetic tree (optional)

This step is optional, the phylogeny tree can either be constructed from representative sequences (e.g. marker gene or amplicon), or provided by users from a NEWICK format file (e.g. for shotgun). The evolutionary tree of the representative sequence is constructed by FastTree and Mafft. The related software can be downloaded to the official website, or please contact my e-mail. The commands involved are as follows:

Mafft (Multiple sequence alignment):

Before performing multiple sequence alignment, the representative sequences of all OTUs need to be found from the GreenGenes database.

```
mafft --auto example/example_respresent_seqs.txt > example/output_aligned.fasta // you need to download Mafft
```

FastTree (Build a phylogenetic tree):

```
FastTree -nt -gtr example/output_aligned.fasta > example/data/example.tree  // you need to download FastTree
```

### Step 2: Get cophenetic distance matrix

The cophenetice distance matrix is obtained by using the cophenetic function in the R package ape. The related software can be downloaded to the official website, or please contact my e-mail.

R script (Get cophenetic distance):

It should be noted that the input and output paths must be absolute paths.

```
usage: PM_CNN/code/get_dis_matrix.R parameter1 parameter2 parameter3

parameter1: current working directory

parameter2: Input a phylogenetic tree file in NEWICK format (This comes from the results generated by FastTree in step 1) 

parameter3: Output cophenetic distance matrix .csv format file
```

#### Example running:

```
cd PM_CNN/code

Rscript get_distance_matrix.R "/home/bioinfo/wangqq/PM_CNN/example" "example.tree" "example_distance_matrix.csv"
```

After obtaining the distance matrix, we aim to obtain the phylogenetic correlation between OTU. Therefore, the next step is to transform the distance matrix into the correlation matrix by the designed distance transformation formula, and then carry out hierarchical clustering based on the correlation matrix, and finally get the result of multi-layer clustering.

### Step 3: Distance transformation and hierarchical clustering

```
usage: code/get_represent_seqs.py 

[--input] //Input cophenetic distance matrix .csv format file (This comes from the results generated by step 2)
[--output] //Output hierarchical clustering results
```

#### Example running:

```
python clustering.py --input /root/user/PM_CNN/example/example_distance_matrix.csv --output /root/user/PM_CNN/example/output_file.csv

```

### Step 4: Model training and prediction

Now, after the processing of Steps 1-3, we now have the clustered feature order. Next, we will start training our model. Before that, we must first introduce the usage of PM-CNN.

### Usage

#### You can see the parameters of PM-CNN through the comment information:

```
usage: model/PMCNN.py [--train] //train PM-CNN model [--test] //test PM-CNN model
                                        [--train_otu_table] //Microbial abundance table
                                        [--meta] //Sample disease information
                                        [--test_otu_table] //test set
                                        [--clustered_groups] //clustering results
                                        [--batch_size] //batch_size
                                        [--learning_rate] //learning rate
                                        [--channel] //number of input and output channels
                                        [--kernel_size] //Convolution kernel size
                                        [--strides] //strides size
```

### Example running:

The training of the model is run using the case microbial sample abundance table under the example file so that readers can quickly get started with PM-CNN.

#### Training:
You can view the training process directly in the console.

```
cd PM_CNN/code

python PMCNN.py --train --train_otu_table ../example/train_data/example_abundance_table.csv --meta ../example/train_data/meta.csv
```

#### Testing:
You can view test results directly in the console.

```
python PMCNN.py --test --test_otu_table ../example/test_data/example_test.csv
```

## Dataset introduction

### [Dataset 1](https://github.com/qdu-bioinfo/PM_CNN/tree/main/data/the%20human%20gut%20microbiome%20(dataset1))


The human oral microbiome data contains 1587 samples with 1554 OTUs. Contains three label types: Control(653), Periodontitis(274), and Gingivitis(660), which are marked with 0, 1, and 2 respectively. 

### [Dataset 2](data/the human oral microbiom (dataset2))

The human gut microbiome data contains 3113 samples with 5597 OTUs. Contains five label types: Control(1418), IBD(993), HIV(360), EDD(222), and CRC(120), which are marked with 0, 1, 2, 3, and 4 respectively.


## Contact

#### All problems please contact PM-CNN development team:**Xiaoquan Su**    Email:[suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
