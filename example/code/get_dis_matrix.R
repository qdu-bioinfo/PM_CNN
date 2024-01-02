# Install and load the optparse package
if (!requireNamespace("optparse", quietly = TRUE)) {
  install.packages("optparse")
}

library(optparse)
library(ape)

# Create option list
option_list <- list(
  make_option(c("--input", "-i"), type = "character", help = "Input tree file"),
  make_option(c("--output", "-o"), type = "character", help = "Output distance matrix file")
)

# Parse command-line arguments
opt_parser <- OptionParser(usage = "Usage: Rscript script.R --input INPUT_FILE --output OUTPUT_FILE", option_list = option_list)
opt <- parse_args(opt_parser)

# Check if both input and output options are provided
if (is.null(opt$input) || is.null(opt$output)) {
  stop("Both --input and --output options are required. See usage with --help.")
}

# Read tree file
tree <- read.tree(opt$input)

# Calculate phylogenetic distances
distances <- cophenetic(tree)

# Write CSV file
write.csv(distances, file = opt$output)
