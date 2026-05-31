import pandas as pd
# from model.classifies import Model

class Predictor:
    def __init__(self):
        pass

    def _missing(self, features, indices):
        """Check if all given indices are missing (-999)"""
        return all(features[i] == -999 for i in indices)

    def predict(self, features):
        # features order:
        # 0=Angle, 1=Long, 2=Long larg, 3=Diameter, 4=Eps,
        # 5=Hauteur, 6=Amorce, 7=Dimension, 8=Developpé, 9=Qte,
        # 10=Diam circle, 11=Larg

        # Missing(expanded_length, amorce, long_larg, dimension, long, larg, circle_diameter)
        if self._missing(features, [8, 6, 2, 7, 1, 11, 10]):
            return "13"

        # Missing(long_larg, dimension, long, larg, circle_diameter)
        elif self._missing(features, [2, 7, 1, 11, 10]):
            return "20"

        # Missing(long_larg, long, larg, circle_diameter)
        elif self._missing(features, [2, 1, 11, 10]):
            return "B"

        # Missing(long_larg, dimension, long, larg, height)
        elif self._missing(features, [2, 7, 1, 11, 5]):
            return "J"

        # Missing(dimension, long, larg, circle_diameter, height)
        elif self._missing(features, [7, 1, 11, 10, 5]):
            return "T"

        return None  # no rule matched, user picks manually