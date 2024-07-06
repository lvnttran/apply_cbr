import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import joblib

import sys
import json

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator KMeans.*")

if len(sys.argv) > 1:
    # Get the argument passed from the other script
    new_value_J_str = sys.argv[1]

    # Convert the JSON string back to a list
    new_value_J = json.loads(new_value_J_str)

    print("Received data:", new_value_J)
    # Continue processing new_value_J as needed
else:
    print("No data received")

# Load Excel file into a pandas ExcelFile object
excel_file = r'P:\CBR\cbr_scr\ver_3\excel\df_sheet_J_nna_nol_tp_dia.xlsx'
pd.set_option('display.max_rows', None)
df_sheetJ = pd.read_excel(excel_file)

kmeans_model_path = r'P:\CBR\cbr_scr\ver_3\model\kmean_J.joblib'
kmeans_loaded = joblib.load(kmeans_model_path)

# Define the order of features (ensure it matches the order used during training)
feature_order = ['developpé', 'amorce', 'angle', 'diam_circle', 'eps', 'diameter', 'Qte']

# Create a list of values in the correct order
new_value_list = [new_value_J[feature] for feature in feature_order]
# Convert the list to a 2D numpy array
new_value_J = np.array(new_value_list)
predicted_cluster = kmeans_loaded.predict([new_value_J])[0]
print(f"Predicted Cluster  : {predicted_cluster}")

# Convert the array back to a dictionary and add the predicted cluster
new_value_J = new_value_J.tolist()
new_value_J = {feature: new_value_J[i] for i, feature in enumerate(feature_order)}

# Add the predicted cluster to the dictionary
new_value_J['Cluster'] = predicted_cluster
print('New data: ', new_value_J)


def df_display(df_name):
    print(df_name.shape)
    print(df_name.head())
    print(df_name.tail(1))


def split_list_to_dict(train_excel_path, test_excel_path):
    """
    Read train and test data from Excel files, and convert them to lists of dictionaries.

    Parameters:
    train_excel_path (str): Path to the Excel file containing the train data.
    test_excel_path (str): Path to the Excel file containing the test data.

    Returns:
    tuple: Two lists of dictionaries containing the train and test data respectively.
    """
    # Load the train and test datasets from Excel files
    df_sheetJ_80 = pd.read_excel(train_excel_path)
    df_sheetJ_20 = pd.read_excel(test_excel_path)

    # Convert DataFrames to lists of dictionaries
    df_sheetJ_80_dict = df_sheetJ_80.to_dict(orient='records')
    df_sheetJ_20_dict = df_sheetJ_20.to_dict(orient='records')

    return df_sheetJ_80_dict, df_sheetJ_20_dict


# Example usage:
train_excel_path = r'P:\CBR\cbr_scr\ver_3\excel\df_sheet_J_nna_nol_tp_dia_train.xlsx'
test_excel_path = r'P:\CBR\cbr_scr\ver_3\excel\df_sheet_J_nna_nol_tp_dia_test.xlsx'

df_sheetJ_80_dict, df_sheetJ_20_dict = split_list_to_dict(train_excel_path, test_excel_path)


def df_to_filtered_dict(df_dict, column_names):
    """
    Convert a list of dictionaries to a filtered list of dictionaries with only specified columns.

    Parameters:
    df_dict (list): List of dictionaries representing the DataFrame rows.
    column_names (list): List of column names to retain in the dictionaries.

    Returns:
    list: A list of filtered dictionaries containing only the specified columns.
    """
    filtered_dict_list = []
    for row_dict in df_dict:
        filtered_row_dict = {key: row_dict[key] for key in column_names if key in row_dict}
        filtered_dict_list.append(filtered_row_dict)
    return filtered_dict_list


column_names = ['developpé', 'amorce', 'angle', 'diam_circle', 'eps', 'diameter', 'Qte']

df_sheetJ_80_dict_modified = df_to_filtered_dict(df_sheetJ_80_dict, column_names)
df_sheetJ_20_dict_modified = df_to_filtered_dict(df_sheetJ_20_dict, column_names)


def dict_to_np_array(data_dict_list, column_names):
    """
    Convert a list of dictionaries to a numpy array.

    Parameters:
    data_dict_list (list): List of dictionaries representing the DataFrame rows.
    column_names (list): List of column names to include in the numpy array.

    Returns:
    numpy.ndarray: A numpy array containing the data from the specified columns.
    """
    return np.array([[item[col] for col in column_names] for item in data_dict_list])


X = dict_to_np_array(df_sheetJ_80_dict_modified, column_names)

# Train the KDTree
kdtree = KDTree(X)


def query_kdtree(kdtree, df, new_value, k):
    # Convert the new value to a numpy array
    new_X = np.array([[new_value['developpé'], new_value['amorce'], new_value['angle'], new_value['diam_circle'],
                       new_value['eps'], new_value['diameter'], new_value['Qte']]])

    # Query the KDTree for nearest neighbors
    distances, indices = kdtree.query(new_X, k=k)

    # Convert the list-like object to a NumPy array
    indices_list = np.array(indices).flatten().tolist()

    # List to store the traced back points
    kdtree_points = []

    # Loop through each index in the indices list
    for index in indices_list:
        # Retrieve the corresponding data point using the index
        kdtree_point = df[index]
        # Append the traced back point to the list
        kdtree_points.append(kdtree_point)

    # Convert the traced back points list to a DataFrame
    kdtree_points_df = pd.DataFrame(kdtree_points)

    return kdtree_points_df


def cal_range_multi(df, column_names):
    """
    Calculate the range of values for multiple columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the columns.
        column_names (list): List of column names for which to calculate the range.

    Returns:
        dict: A dictionary where keys are range_column_name and values are the corresponding column names.
    """
    ranges = {}
    for col_name in column_names:
        # print(df[col_name].max())
        # print(df[col_name].min())
        column_range = round(df[col_name].max() - df[col_name].min(), 1)
        if column_range == 0:
            column_range = 1  # Replace zero range with 1
        ranges[col_name] = column_range
    return ranges


weight_set = [2, 3, 3, 3, 1, 1, 3]


def weight(df, column_names, weight_set):
    weights = {}
    for i, col_name in enumerate(column_names):
        weights[col_name] = weight_set[i]
    return weights


def calculate_similarity(df_list, ranges_dict, weights_dict, weight_set, column_names, new_value, top_n):
    similarities = []

    for row in range(len(df_list)):
        ls_line = {}
        glo_sumproduct = 0

        for col in column_names:
            local_similarity = round(1 - (abs(new_value[col] - df_list[row][col]) / ranges_dict[col]), 2)
            ls_line[col] = local_similarity
            glo_sumproduct += weights_dict[col] * local_similarity

        glo_similarity = round(((1 / sum(weight_set)) * glo_sumproduct), 2)

        similarities.append((row, ls_line, glo_similarity))

    # Sort similarities based on global similarity (GS)
    similarities_sorted = sorted(similarities, key=lambda x: x[2], reverse=True)

    # Get the top N rows
    top_n_rows = similarities_sorted[:top_n]

    # Calculate the mean of the last_item_value for the top N rows
    last_item_values = [df_sheetJ_80_dict[row[0]][list(df_sheetJ_80_dict[row[0]].keys())[-1]] for row in top_n_rows]
    # print(last_item_values)
    mean_last_item_value = round(sum(last_item_values) / len(last_item_values), 2)

    # Add the mean last_item_value to the new_value dictionary using the same key as the original last_item
    last_item_key = list(df_sheetJ_80_dict[top_n_rows[0][0]].keys())[-1]
    new_value[last_item_key] = mean_last_item_value

    return top_n_rows, mean_last_item_value, new_value


df_kdtree = query_kdtree(kdtree, df=df_sheetJ_80_dict, new_value=new_value_J, k=350)

ranges_dict = cal_range_multi(df_kdtree, column_names)
print("Range: ", ranges_dict)

weights_dict = weight(df_kdtree, column_names, weight_set)
print("Weight: ", weights_dict)

top_n_rows, mean_last_item_value, updated_new_value = calculate_similarity(
    df_sheetJ_80_dict,
    ranges_dict,
    weights_dict, weight_set,
    column_names, new_value_J, top_n=2)

# Store the results in a dictionary
result = {
    "Top Rows": top_n_rows,
    "Predicted Value": mean_last_item_value,
    "Updated New Value": updated_new_value
}

print("Full params:", result["Updated New Value"])
print("Predict time is:", result["Predicted Value"])
