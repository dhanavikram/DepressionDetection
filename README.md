# Evaluation of Gender Bias in Depression Detection Models

**Authors:**

- Kalyani Jaware
- Erin Richardson
- Dayn Reoh
- Dhanavikram Sekar

Digital healthcare is increasingly leveraging speech-based ML technologies due to the ease of unobtrusive data collection through smartphones and wearables. Speech carries valuable insights into human behavior and mental state, stemming from the complex interplay of cognitive planning and articulation. Acoustic measures from speech, like prosody, provide critical information for mental healthcare, but may also be confounded by demographic factors, potentially leading to biases in ML algorithms detecting depression.

In this project, we build predictive models of depression and gender. We also explore and remove gender bias in these models and compare their performance to that of their prior versions.

## Detailed Explanation of Methodologies:

This [report](Final Report.pdf) explains the methodologies followed to detect Depression in humans and the filter-based techniques used to evaluate the effect of gender in model performances.

## TL;DR

Take a look at this [poster](E-poster.pdf)


# Folder Structure

    .
    ├── data                            # Original data files
    ├── depression_classification       # Notebooks used for the classification of depression
    ├── depression_subset_selection     # Notebooks used to find and remove depression-identifying features
    ├── depression_classification       # Notebooks used for the classification of gender
    ├── depression_subset_selection     # Notebooks used to find and remove gender-identifying features
    ├── models                          # Initial models attempted 
    ├── outputs                         # Output CSV files
    ├── pictures                        # Output image files
    ├── processed_data                  # Intermediate transformed CSV files
    ├── 00_data_read.ipynb              # Notebook to read data and merge different CSV files
    ├── 01_clean_data.ipynb             # Notebook to clean data and compute principal components
    ├── 02_feature_selection.ipynb      # Notebook to select feat. based on correlation and fishers' criterion
    └── README.md


This work is a part of the CSCI-5622 Class (Spring 2024).