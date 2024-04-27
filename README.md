# To-Do:
- Atleast two classification algorithms for Depression and Gender Classification
    - Report accuracy on the test set (on participant level)
    - Report balanced accuracy on test set (on participant level)
    - Report equality of opportunity (EO) (on participant level) (only for depression classification)
    - Discuss the results
- Find informative features for depression (filter methods)
    - Select n features (n = 10 to 50, step size = 10(flexible))
    - Run the model on n features and compare it to original model (Use same algo)
    - Plot accuracy, balanced accuracy and EO of different models (n vs metric)
- Find informative features for gender (filter methods)
    - Select m features (m = 10 to 50, step size = 10(flexible))
    - Run the model on m features and compare it to original model (Use same algo)
    - Plot accuracies and balanced accuracies of different models (m vs metric)
- Remove m most informative features of gender from feature set and compare it with og model
    - Report the findings
- Optional (other bias mitigation approaches)
- E-poster


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