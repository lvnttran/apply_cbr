import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import joblib
import json

class KdCbrBase():
    def __init__(self, config):
        excel_file = config['excel_file']
        kmeans_model_path = config['kmeans_model_path']
        train_excel_path = config['train_excel_path']
        test_excel_path = config['test_excel_path']
        self.feature_order = config['feature_order']
        self.column_names = config['feature_order'] + ['Cluster']
        self.weight_set = [1 for i in range(len(self.feature_order))] + [3]

        pd.set_option('display.max_rows', None)
        self.df_sheetOT = pd.read_excel(excel_file)
        self.kmeans_loaded = joblib.load(kmeans_model_path)

        self.df_sheet_80_dict, _ = self.split_list_to_dict(train_excel_path, test_excel_path)
        df_sheet_80_dict_modified = self.df_to_filtered_dict(self.df_sheet_80_dict, self.column_names)
        X = self.dict_to_np_array(df_sheet_80_dict_modified, self.column_names)
        self.kdtree = KDTree(X)
    
    def predict_cluster(self, value_list):
        new_value = {}
        for i, va in enumerate(self.feature_order):
            new_value[va] = value_list[i]
        # Create a list of values in the correct order
        new_value_list = [new_value[feature] for feature in self.feature_order]
        # Convert the list to a 2D numpy array
        new_value = np.array(new_value_list)
        predicted_cluster = self.kmeans_loaded.predict([new_value])[0]
        return predicted_cluster
    
    def predict_time(self, value_list, k=200, n=5):
        try:
            new_value = {}
            for i, va in enumerate(self.feature_order):
                new_value[va] = value_list[i]
            new_value['Cluster'] = value_list[-1]
            df_kdtree = self.query_kdtree(self.kdtree, df=self.df_sheet_80_dict, new_value=new_value, k=k)
            ranges_dict = self.cal_range_multi(df_kdtree, self.column_names)
            weights_dict = self.weight(df_kdtree, self.column_names, self.weight_set)
            top_n_rows, mean_last_item_value, updated_new_value = self.calculate_similarity(
                self.df_sheet_80_dict,
                ranges_dict,
                weights_dict, self.weight_set,
                self.column_names, new_value, top_n=n, df_sheetJ_80_dict = self.df_sheet_80_dict)
            return True, {
                "Top Rows": top_n_rows,
                "Predicted Value": mean_last_item_value,
                "Updated New Value": updated_new_value
            }
        except Exception as e:
            return False, str(e)

    def df_display(self, df_name):
        print(df_name.shape)
        print(df_name.head())
        print(df_name.tail(1))

    def split_list_to_dict(self, train_excel_path, test_excel_path):
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

    def df_to_filtered_dict(self, df_dict, column_names):
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
    
    def dict_to_np_array(self, data_dict_list, column_names):
        """
        Convert a list of dictionaries to a numpy array.

        Parameters:
        data_dict_list (list): List of dictionaries representing the DataFrame rows.
        column_names (list): List of column names to include in the numpy array.

        Returns:
        numpy.ndarray: A numpy array containing the data from the specified columns.
        """
        return np.array([[item[col] for col in column_names] for item in data_dict_list])
    
    def query_kdtree(self, kdtree, df, new_value, k):
        # Convert the new value to a numpy array
        new_value_array = [new_value[i] for i in new_value]
        new_X = np.array([new_value_array])
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

    def cal_range_multi(self, df, column_names):
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
    
    def weight(self, df, column_names, weight_set):
        weights = {}
        for i, col_name in enumerate(column_names):
            weights[col_name] = weight_set[i]
        return weights

    def calculate_similarity(self, df_list, ranges_dict, weights_dict, weight_set, column_names, new_value, top_n, df_sheetJ_80_dict):
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

    