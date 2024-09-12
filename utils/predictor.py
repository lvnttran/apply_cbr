import pandas as pd
from model.classifies import Model

class Predictor():
    def __init__(self):
        self.model = Model(r'src\models\nn__epoch_1434_acc_0.9602.h5') #r'src\models\nn__epoch_1434_acc_0.9602.h5'

    # Function to create a dictionary from values and column names
    def create_value_dict(self, value_set, column_names):
        new_column_names = ['diam_circle', 'larg', 'eps', 'Qte', 'diameter', 'hauteur',
                      'long larg', 'developpé', 'angle', 'amorce', 'long', 'Dimension' ]
        value_dict = {col_name: [value_set[column_names.index(col_name)]] for i, col_name in enumerate(new_column_names)}
        return value_dict

    def extract_cbr_values(self, input_data, column_names):
        extracted_values = {col: float(input_data[col][0]) for col in column_names}
        return extracted_values

    def process_data(self, value_set) -> pd.DataFrame:
        selected_columns = ['angle', 'long', 'long larg', 'diameter', 'eps', 'hauteur',
                            'amorce', 'Dimension', 'developpé', 'Qte', 'diam_circle', 'larg']
        value_dict = self.create_value_dict(value_set, selected_columns)
        return pd.DataFrame(value_dict)

    def predict(self, value_set):
        processed_data = self.process_data(value_set)
        predicted_label = self.model.predict(processed_data)
        return predicted_label

# if __name__ == '__main__':
#     predictor = Predictor()
#     value_set = [89.954374, -999, -999, 1600.0, 2.5, -999.0, 0, -999, 1256.0, 8, 42.4, -999]
#     print(predictor.predict(value_set))



# Example values for prediction
# # Conditional logic based on predicted label
# if predicted_label == '13':
#     subprocess.run(['python', 'kd_cbr_13.py'])
# elif predicted_label == '20':
#     subprocess.run(['python', 'kd_cbr_20.py'])
# elif predicted_label == 'B':
#     subprocess.run(['python', 'kd_cbr_B.py'])
# elif predicted_label == 'G':
#     subprocess.run(['python', 'kd_cbr_G.py'])
# elif predicted_label == 'J':
#     column_names = ['developpé', 'amorce', 'angle', 'diam_circle', 'eps', 'diameter', 'Qte']
#     new_value_J = extract_values_cbr(new_value, column_names)
#     print('Sent data: ', new_value_J)
#     new_value_J_str = json.dumps(new_value_J)
#     subprocess.run(['python', 'kd_cbr_J.py', new_value_J_str])
# elif predicted_label == 'M':
#     subprocess.run(['python', 'kd_cbr_M.py'])
# elif predicted_label == 'OT':
#     subprocess.run(['python', 'kd_cbr_OT.py'])
# elif predicted_label == 'T':
#     subprocess.run(['python', 'kd_cbr_T.py'])
# else:
#     print("No specific action defined for this label.")