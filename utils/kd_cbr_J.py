from utils.kd_cbr_base import *

class KdCbrJ(KdCbrBase):
    def __init__(self, feature_order):
        KdCbrBase.__init__(self)
        # Load Excel file into a pandas ExcelFile object
        excel_file = r'excel\df_sheet_J_nna_nol_tp_dia.xlsx'
        pd.set_option('display.max_rows', None)
        self.df_sheetJ = pd.read_excel(excel_file)

        kmeans_model_path = r'src\models\kmean_J.joblib'
        self.kmeans_loaded = joblib.load(kmeans_model_path)

        train_excel_path = r'excel\df_sheet_J_nna_nol_tp_dia_train.xlsx'
        test_excel_path = r'excel\df_sheet_J_nna_nol_tp_dia_test.xlsx'

        self.column_names = ['developpé', 'amorce', 'angle', 'diam_circle', 'eps', 'diameter', 'Qte', 'Cluster']
        self.feature_order = feature_order


        self.df_sheetJ_80_dict, self.df_sheetJ_20_dict = self.split_list_to_dict(train_excel_path, test_excel_path)
        df_sheetJ_80_dict_modified = self.df_to_filtered_dict(self.df_sheetJ_80_dict, self.column_names)
        # df_sheetJ_20_dict_modified = self.df_to_filtered_dict(df_sheetJ_20_dict, column_names)


        X = self.dict_to_np_array(df_sheetJ_80_dict_modified, self.column_names)

        # Train the KDTree
        self.kdtree = KDTree(X)

        self.weight_set = [1, 1, 1, 1, 1, 1, 1, 3]
   
    def predict_cluster(self, new_value_J_list: list):
        new_value_J = {}
        for i, va in enumerate(self.column_names):
            new_value_J[va] = new_value_J_list[i]

        # Create a list of values in the correct order
        new_value_list = [new_value_J[feature] for feature in self.feature_order]
        # Convert the list to a 2D numpy array
        new_value_J = np.array(new_value_list)
        predicted_cluster = self.kmeans_loaded.predict([new_value_J])[0]
        print(f"Predicted Cluster  : {predicted_cluster}")
        return predicted_cluster
    
    def predict_time(self, new_value_J_list, k = 400, n = 5):
        new_value_J = {}
        for i, va in enumerate(self.column_names):
            new_value_J[va] = new_value_J_list[i]
        new_value_J['Cluster'] = new_value_J_list[-1]
        print('new_value_J', new_value_J)
        df_kdtree = self.query_kdtree(self.kdtree, df=self.df_sheetJ_80_dict, new_value=new_value_J, k=k)

        ranges_dict = self.cal_range_multi(df_kdtree, self.column_names)
        print("Range: ", ranges_dict)

        weights_dict = self.weight(df_kdtree, self.column_names, self.weight_set)
        print("Weight: ", weights_dict)

        top_n_rows, mean_last_item_value, updated_new_value = self.calculate_similarity(
            self.df_sheetJ_80_dict,
            ranges_dict,
            weights_dict, self.weight_set,
            self.column_names, new_value_J, top_n=n, df_sheetJ_80_dict= self.df_sheetJ_80_dict)

        # Store the results in a dictionary
        result = {
            "Top Rows": top_n_rows,
            "Predicted Value": mean_last_item_value,
            "Updated New Value": updated_new_value
        }
        return result

