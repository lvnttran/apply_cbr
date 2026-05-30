import sys

print(sys.executable)
print(sys.path)

from model.classifies import Model
model = Model(r'src\models\nn__epoch_1434_acc_0.9602.h5')
print("OK")