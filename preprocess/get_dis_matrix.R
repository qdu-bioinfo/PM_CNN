setwd("/save_path")
install.packages("ape")
                      
library(ape)

tree <- read.tree("Gut.tree")

distances <- cophenetic(tree)

write.csv(distances,"distance_matrix.csv")