from utils.kd_cbr_base import *

class KdCbrOT(KdCbrBase):
    def __init__(self):
        KdCbrBase.__init__(self)
        # Load Excel file into a pandas ExcelFile object
        excel_file = r'excel/df_sheet_OT_nna_nol_tp_dia.xlsx'
        pd.set_option('display.max_rows', None)
        self.df_sheetOT = pd.read_excel(excel_file)

        kmeans_model_path = r'src/models/kmean_OT.joblib'
        self.kmeans_loaded = joblib.load(kmeans_model_path)

        # Define the order of features (ensure it matches the order used during training)
        self.feature_order = ['Qte', 'diameter', 'developpé', 'amorce', 'angle', 'long larg', 'eps', 'Dimension', 'long',
                        'larg',
                        'diam_circle']
        self.column_names = ['Qte', 'diameter', 'developpé', 'amorce', 'angle', 'long larg', 'eps', 'Dimension', 'long',
                            'larg',
                            'diam_circle', 'Cluster']
        
        # Example usage:
        train_excel_path = r'excel/df_sheet_OT_nna_nol_tp_dia_k_train.xlsx'
        test_excel_path = r'excel/df_sheet_OT_nna_nol_tp_dia_k_test.xlsx'
        self.df_sheetOT_80_dict, _ = self.split_list_to_dict(train_excel_path, test_excel_path)
        df_sheetOT_80_dict_modified = self.df_to_filtered_dict(self.df_sheetOT_80_dict, self.column_names)
       
        X = self.dict_to_np_array(df_sheetOT_80_dict_modified, self.column_names)
        # Train the KDTree
        self.kdtree = KDTree(X)
        self.weight_set = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]

        
    def predict_cluster(self, new_value_OT_list):
            new_value_OT = {}
            for i, va in enumerate(self.column_names):
                new_value_OT[va] = new_value_OT_list[i]
            # Create a list of values in the correct order
            new_value_list = [new_value_OT[feature] for feature in self.feature_order]
            # Convert the list to a 2D numpy array
            new_value_OT = np.array(new_value_list)
            predicted_cluster = self.kmeans_loaded.predict([new_value_OT])[0]
            return predicted_cluster
    
    def predict_time(self, new_value_OT_list, k=200, n=5):
            new_value_OT = {}
            for i, va in enumerate(self.column_names):
                new_value_OT[va] = new_value_OT_list[i]
            new_value_OT['Cluster'] = new_value_OT_list[-1]


            df_kdtree = self.query_kdtree(self.kdtree, df=self.df_sheetOT_80_dict, new_value=new_value_OT, k=k)

            ranges_dict = self.cal_range_multi(df_kdtree, self.column_names)
            print("Range: ", ranges_dict)

            weights_dict = self.weight(df_kdtree, self.column_names, self.weight_set)
            print("Weight: ", weights_dict)

            top_n_rows, mean_last_item_value, updated_new_value = self.calculate_similarity(
                self.df_sheetOT_80_dict,
                ranges_dict,
                weights_dict, self.weight_set,
                self.column_names, new_value_OT, top_n=n)

            # Store the results in a dictionary
            result = {
                "Top Rows": top_n_rows,
                "Predicted Value": mean_last_item_value,
                "Updated New Value": updated_new_value
            }

            return result


