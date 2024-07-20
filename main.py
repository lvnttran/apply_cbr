from tum_gui import *
import sys
import yaml

from utils.predictor import Predictor
from utils.kd_cbr_base import KdCbrBase


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.predictor = Predictor()
        self.kd_cbrs = {}

        self.ui.btn_data_processing.clicked.connect(self.processing_data)
        self.ui.btn_autolabel.clicked.connect(self.predict_data)
        self.ui.btn_calculate2.clicked.connect(self.predict_data_cbr)
        self.ui.btn_calculate_12.clicked.connect(self.calculate_form1)
        self.ui.btn_data_processing2.clicked.connect(lambda x: self.processing_form2())
        self.load_kd_cbr()

    def load_kd_cbr(self):
        for config in self.ui.config:
            self.kd_cbrs[config] = KdCbrBase(self.ui.config[config])

    def processing_data(self):
        lineEdits = self.ui.get_line_edit_select()
        if lineEdits is None:
            QtWidgets.QMessageBox.information(self, "Warning", f"Please select row!",
                                              QtWidgets.QMessageBox.StandardButton.Ok)
        if lineEdits is not None:
            self.ui.add_label_to_log_programs(text='Processing data...')
            for name, line_edit in lineEdits:
                if line_edit.text() == "":
                    line_edit.setText('-999')
                    self.ui.add_label_to_log_programs(text=f'Reset {name} value: -999')
            self.ui.add_label_to_log_programs(type=0, text='Processing completed!')
            return True

    def processing_form2(self):
        self.ui.procesing_data_form2()
        self.predict_cluster()

    def predict_cluster(self):
        lineEdits = self.ui.get_value_line_edit_form2()
        if lineEdits is not None:
            label = self.ui.comboBox_label2.currentText()
            if label != "None":
                predictor = self.kd_cbrs[label]
                print('predictor', predictor)
                cluster = predictor.predict_cluster([float(i) for i in lineEdits])
                self.ui.add_label_to_log_programs(type=0, text=f'Cluster: {cluster}')
                self.ui.lineEdit_3.setText(str(cluster))

    def predict_data(self):
        self.processing_data()
        lineEdits = self.ui.get_line_edit_select()
        if lineEdits is not None:
            data = [float(lineEdit.text()) for _, lineEdit in lineEdits]
            kq = self.predictor.predict(data)
            kq = kq if kq in self.ui.data else "None"
            self.ui.comboBox_label.setCurrentText(kq)
            self.ui.add_label_to_log_programs(type=0, text=f'Result: {kq}')

    def calculate_form1(self):
        if self.processing_data():
            label = self.ui.comboBox_label.currentText()
            if label in self.ui.data_keys:
                indexs = [self.ui.values.index(key) for key in self.ui.data_keys[label]]
                lineEdits = self.ui.get_line_edit_select()
                data = [float(lineEdit.text()) for _, lineEdit in lineEdits]
                values = [self.ui.format_number(data[index]) for index in indexs]
                self.set_calculate_form2(label, values)
                return
            else:
                QtWidgets.QMessageBox.warning(self, 'Invalid Type', 'Please select a valid type!')
                return None

    def set_calculate_form2(self, label, values):
        self.ui.comboBox_label2.setCurrentText(label)
        lineEdits = self.ui.get_line_edit_form_2()
        for i, lineEdit in enumerate(lineEdits):
            lineEdit.setText(str(values[i]))
        self.ui.tabWidget.setCurrentIndex(1)

    def predict_data_cbr(self):
        self.ui.procesing_data_form2()
        self.ui.procesing_data_kn()
        self.predict_cluster()
        lineEdits = self.ui.get_value_line_edit_form2()
        if lineEdits is not None:
            k = int(self.ui.edt_k.text())
            n = int(self.ui.edt_n.text())
            lineEdits.append(float(self.ui.lineEdit_3.text()))
            label = self.ui.comboBox_label2.currentText()
            predictor = self.kd_cbrs[label]
            is_, dict_value = predictor.predict_time(lineEdits, k, n)
            if is_:
                top_row = dict_value['Top Rows']
                predicted_value = dict_value['Predicted Value']
                updated_new_value = dict_value['Updated New Value']
                self.ui.label_6.setText(f'Predicted Value: {predicted_value}')
                self.ui.add_label_to_log_programs(type=0, text=f'Top Rows: {top_row}')
                self.ui.add_label_to_log_programs(type=0, text=f'Predicted Value: {predicted_value}')
                self.ui.add_label_to_log_programs(type=0, text=f'Updated New Value: {updated_new_value}')
            else:
                QtWidgets.QMessageBox.information(self, "Warning", dict_value, QtWidgets.QMessageBox.StandardButton.Ok)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ui = GUI()
    ui.show()
    sys.exit(app.exec_())
