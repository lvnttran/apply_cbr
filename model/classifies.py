from tensorflow.keras.models import load_model
import numpy as np

class Model:
    def __init__(self, model_path):
        self.loaded_model = load_model(model_path)

    def predict(self, new_value):
        predictions = self.loaded_model.predict(new_value)

        # Get the predicted label (class with highest probability)
        predicted_label_index = np.argmax(predictions, axis=1)
       
        # Example mapping or label encoder usage:
        #label_mapping = {0: '13', 1: '20', 2: 'B', 3: 'G', 4: 'J', 5: 'M', 6: 'OT', 7: 'T'}
        label_mapping = {0: 'OT', 1: '20', 2: '13', 3: 'T', 4: 'J', 5: 'G', 6: 'M', 7: 'B'}

        # Get the actual label corresponding to the predicted index
        predicted_label = label_mapping[predicted_label_index[0]]
        return predicted_label
    

    