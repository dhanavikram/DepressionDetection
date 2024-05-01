import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


# Paths of folders containing data
TRAIN_FOLDER_PATH = "../data/features_train"
TEST_FOLDER_PATH = "../data/features_test"
FEATURE_DESC_PATH = "../data/feature_description.csv"
LABELS_PATH = "../data/labels.csv"


def calculate_class_weights(y, target_col='Gender'):
    """
    Calculates class weight to solve class imbalance problem
    """
    unique_classes, class_counts = np.unique(y[target_col], return_counts=True)
    total_samples = y.shape[0]
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight

    return class_weights


def compute_metrics(y):
    """
    Computes accuracy, balanced accuracy and equality of opportunity for the given result dataframe.
    The input dataframe should contain:
    1. Gender label
    2. Depression label
    3. Predicted Depression label
    4. Participant ID
    """
    
    # Sets target as Gender
    target = "Gender"

    # Aggregate target and predicted labels at participant level 
    y_actual = y.groupby("Participant_ID").agg({target:"max"})[target]
    y_pred = y.groupby("Participant_ID").agg(Predicted=(target+'_predicted', lambda x: x.mode()[0]))["Predicted"]

    # Compute metrics
    accuracy = accuracy_score(y_actual, y_pred)
    bal_accuracy = balanced_accuracy_score(y_actual, y_pred)

    return accuracy, bal_accuracy
    

def show_conf_matrix(y, target="Gender"):
    # Aggregate target and predicted labels at participant level 
    y_actual = y.groupby("Participant_ID").agg({target:"max"})[target]
    y_pred = y.groupby("Participant_ID").agg(Predicted=(target+'_predicted', lambda x: x.mode()[0]))["Predicted"]

    conf_matrix = confusion_matrix(y_actual, y_pred)
    #Visualizing the confusion matrix
    plt.rcParams['figure.figsize'] = (6, 6)
    display_c_m = ConfusionMatrixDisplay(conf_matrix)
    display_c_m.plot(cmap='YlOrBr', colorbar=False)
    # plt.title('Confusion Matrix')
    plt.show()


def read_conv(train=True, cols_to_remove=None):

    if train:
        folder_path = TRAIN_FOLDER_PATH
    else:
        folder_path = TEST_FOLDER_PATH

    # Names of features based on GeMAPS feature set
    meta_data = pd.read_csv(FEATURE_DESC_PATH, encoding='ISO-8859-1', header=None)
    col_names = list(meta_data[0])

    df_dict = {'Participant_ID': [], 'features': []}  # Dict to store values
    max_rows = 0  # Max number of rows present in the data (used for padding)

    for file in os.listdir(folder_path):
        
        file_name_split = file.split('.')
        file_type = file_name_split[1]
        file_name = file_name_split[0]

        if file_type == 'csv':
            # Fetch participant ID
            id = int(file_name.split('_')[1])
            df_dict['Participant_ID'].append(id)

            # Fetch data
            temp_df = pd.read_csv(folder_path + '/' + file, names=col_names)

            # Drop columns with high collinearity
            if cols_to_remove is not None:
                temp_df = temp_df.drop(cols_to_remove, axis=1)

            # Remove null values
            if temp_df.isna().sum().values[0]>0:
                print(f"Removing null values present in {file}")
                temp_df = temp_df.dropna(axis=0)
            
            # Filter out rows where more than half of the feature values are zero
            zero_percentages = (temp_df == 0).mean(axis=1)  # Calculate the percentage of zero values in each row
            threshold = 0.5  # More than half
            temp_df = temp_df[zero_percentages <= threshold]
            
            # Add the features to dict
            df_dict['features'].append(temp_df)
            
            # Update max rows
            if temp_df.shape[0]>max_rows:
                max_rows = temp_df.shape[0] 
            
    num_features = list(df_dict.values())[1][0].shape[1]
    
    return df_dict, max_rows, num_features


def prep_conv_data(df_dict, max_rows, num_features):

    # Pad zeroes
    for i in range(len(df_dict['features'])):
        df_dict['features'][i] = df_dict['features'][i].reindex(np.arange(max_rows), fill_value=0).values
    df_features = pd.DataFrame(df_dict)

    # Read labels dict
    df_labels = pd.read_csv(LABELS_PATH, skipfooter=1, engine='python')
    df_labels['Participant_ID'] = df_labels['Participant_ID'].astype(int)
    merged_df = pd.merge(df_features, df_labels, on='Participant_ID')

    # Extract Features and Labels for tensorflow
    features = merged_df['features'].to_list()
    depression_labels = merged_df['Depression'].values
    gender_labels = merged_df['Gender'].values

    # Convert features to tensor object
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    features = np.array(features_tensor).reshape(depression_labels.size, max_rows, num_features, 1)

    # Convert labels to proper shape
    depression_labels = np.asarray(depression_labels).astype('float32').reshape((-1,1))
    gender_labels = np.asarray(gender_labels).astype('float32').reshape((-1,1))

    return features, depression_labels, gender_labels


def prep_lstm_data(df_dict, max_rows):
    # Read labels dict
    df_labels = pd.read_csv(LABELS_PATH, skipfooter=1, engine='python')
    df_labels['Participant_ID'] = df_labels['Participant_ID'].astype(int)
    
    merged_df = pd.merge(pd.DataFrame(df_dict), df_labels, on='Participant_ID')

    padded_sequences = [np.pad(seq, ((0, max_rows - len(seq)), (0, 0)), mode='constant') for seq in merged_df['features']]
    X = np.array(padded_sequences)

    depression_labels = merged_df['Depression'].values
    gender_labels = merged_df['Gender'].values
    
    # Convert labels to proper shape
    depression_labels = np.asarray(depression_labels).astype('float32').reshape((-1,1))
    gender_labels = np.asarray(gender_labels).astype('float32').reshape((-1,1))

    return X, depression_labels, gender_labels