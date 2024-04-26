# Timeline
* April 17-April 21: Create combined CSV, perform data cleaning and preprocessing
* April 22-April 27: Perform model building, compare different modeling results for classifying both gender and depression
* April 28-29: look at mitigating bias and most informative features
* April 30-May 1: make poster and report

# Tasks
* Erin: making clean data script, transformations, run PCA (generate scree plot to pick # of components), regularization
* Dayn: assessing correlations between variables, building feature set not associated with gender, criteria for informative feature set
* Dhanavikram: work on padding CSVs to facilitate CNNs, model optimization (finetuning hyperparameters)
* Kalyani: trying fully connected networks and CNNs, random forests


# Folder Structure

    .
    ├── data                            # Original data files
    ├── processed_data                  # Cleaned and transformed data files in csv format
    ├── models                          # Notebooks separated via models
    ├── outputs                         # Output images and csv files
    ├── 00_data_read.ipynb              # Read data and merge different csv files
    ├── 01_clean_data.ipynb             # Clean data and compute principal components
    ├── 02_feature_selection.ipynb      # Select features based on correlation and fishers criterion
    └── README.md
