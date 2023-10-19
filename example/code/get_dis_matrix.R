setwd("/save_path")
install.packages("ape")
                      
library(ape)

tree <- read.tree("ex.tree")

distances <- cophenetic(tree)

write.csv(distances,"distance_matrix.csv")