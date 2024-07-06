import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import json



# Function to create a dictionary from values and column names
def get_value(value_set, column_names):
    new_value = {}
    for i, col_name in enumerate(column_names):
        new_value[col_name] = [value_set[i]]  # Ensure each value is in a list format
    return new_value


def extract_values_cbr(input_data, column_names):
    extracted_values = {}
    for col in column_names:
        extracted_values[col] = float(input_data[col][0])  # Convert to standard Python int
    return extracted_values


# Example values for prediction
value_set = [89.954374, -999, -999, 1600.0, 2.5, -999.0, 0, -999, 1256.0, 8, 42.4, -999]
selected_columns_X = ['angle', 'long', 'long larg', 'diameter', 'eps', 'hauteur',
                      'amorce', 'Dimension', 'developpé', 'Qte', 'diam_circle', 'larg']

# Create a DataFrame from the values and column names
new_value = pd.DataFrame(get_value(value_set, selected_columns_X))

# Load the saved neural network model
model_path = r"P:\CBR\cbr_scr\ver_3\model\nn_J.h5"
loaded_model = load_model(model_path)

# Make predictions with the loaded neural network model
predictions = loaded_model.predict(new_value)

# Get the predicted label (class with highest probability)
predicted_label_index = np.argmax(predictions, axis=1)

# Define a mapping or use label encoder to get the actual label from the index
# Example mapping or label encoder usage:
label_mapping = {0: '13', 1: '20', 2: 'B', 3: 'G', 4: 'J', 5: 'M', 6: 'OT', 7: 'T'}

# Get the actual label corresponding to the predicted index
predicted_label = label_mapping[predicted_label_index[0]]

print("Predicted Label:", predicted_label)

# Conditional logic based on predicted label
if predicted_label == '13':
    subprocess.run(['python', 'kd_cbr_13.py'])
elif predicted_label == '20':
    subprocess.run(['python', 'kd_cbr_20.py'])
elif predicted_label == 'B':
    subprocess.run(['python', 'kd_cbr_B.py'])
elif predicted_label == 'G':
    subprocess.run(['python', 'kd_cbr_G.py'])
elif predicted_label == 'J':
    column_names = ['developpé', 'amorce', 'angle', 'diam_circle', 'eps', 'diameter', 'Qte']
    new_value_J = extract_values_cbr(new_value, column_names)
    print('Sent data: ', new_value_J)
    new_value_J_str = json.dumps(new_value_J)
    subprocess.run(['python', 'kd_cbr_J.py', new_value_J_str])
elif predicted_label == 'M':
    subprocess.run(['python', 'kd_cbr_M.py'])
elif predicted_label == 'OT':
    subprocess.run(['python', 'kd_cbr_OT.py'])
elif predicted_label == 'T':
    subprocess.run(['python', 'kd_cbr_T.py'])
else:
    print("No specific action defined for this label.")
