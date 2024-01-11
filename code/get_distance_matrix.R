# Save this code in a file, for example, run_script.R

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if the required arguments are provided
if (length(args) != 3) {
  stop("Usage: Rscript run_script.R <working_directory> <input_tree_file> <output_distance_matrix_file>")
}

# Set working directory
setwd(args[1])

# Install required package
if (!requireNamespace("ape", quietly = TRUE)) {
  install.packages("ape")
}

# Load required library
library(ape)

# Read input tree file and output distance matrix file from command-line arguments
input_tree_file <- args[2]
output_distance_matrix_file <- args[3]

# Read tree file
tree <- read.tree(input_tree_file)

# Calculate cophenetic distances
distances <- cophenetic(tree)

# Write distances to CSV file
write.csv(distances, output_distance_matrix_file)

# Print a message indicating successful execution
cat("Cophenetic distances calculated and saved to", output_distance_matrix_file, "\n")
