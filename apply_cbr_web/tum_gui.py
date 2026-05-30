#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.predictor import Predictor
from utils.kd_cbr_base import KdCbrBase

import sys
from PyQt5 import QtWidgets, QtCore
import yaml



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CBR")
        self.resize(1200, 800)

        self.predictor = Predictor()
        self.kd_cbrs = {}
        self.config = self.load_config()

        self.load_kd_cbr()
        self._setup_ui()

    # ---------------- CONFIG ----------------
    def load_config(self):
        with open('src/config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_kd_cbr(self):
        for label, cfg in self.config.items():
            self.kd_cbrs[label] = KdCbrBase(cfg)

    # ---------------- NORMALIZATION ----------------
    def norm(self, s):
        return str(s).replace('_', ' ').replace('é', 'e').strip().lower()

    # ---------------- UI ----------------
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.main_layout = QtWidgets.QVBoxLayout(central)

        # =========================
        # ROW 1 — INPUT + PROCESS
        # =========================
        self.feature_display = {
            "Angle": "Angle",
            "Long": "Length",
            "Long larg": "Long Large",
            "Diameter": "Diameter",
            "Eps": "Thickness",
            "Hauteur": "Height",
            "Amorce": "Amorce",
            "Dimension": "Dimension",
            "Developpé": "Expanded Length",
            "Qte": "Quantity",
            "Diam circle": "Circle Diameter",
            "Larg": "Width"
        }

        self.features = [
            "Angle", "Long", "Long larg", "Diameter",
            "Eps", "Hauteur", "Amorce", "Dimension",
            "Developpé", "Qte", "Diam circle", "Larg"
        ]

        self.table_input = QtWidgets.QTableWidget(1, len(self.features))
        self.table_input.setHorizontalHeaderLabels(
            [self.feature_display.get(f, f) for f in self.features]
        )
        self.table_input.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        self.btn_process = QtWidgets.QPushButton("Standardize")
        self.btn_process.setFixedHeight(30)
        self.btn_process.setStyleSheet("font-size:15px")

        row1 = QtWidgets.QVBoxLayout()
        row1.addWidget(QtWidgets.QLabel("New ETO Product (NPj) feature-values"))
        row1.addWidget(self.table_input)
        row1.addWidget(self.btn_process)

        self.main_layout.addLayout(row1)

        # =========================
        # ROW 2A — PREDICT LABEL (CLEANED)
        # =========================
        row2a = QtWidgets.QHBoxLayout()

        self.btn_predict = QtWidgets.QPushButton("RBL identify product family")
        self.btn_predict.setFixedHeight(30)
        self.btn_predict.setStyleSheet("font-size:15px")

        self.combo_label = QtWidgets.QComboBox()
        self.combo_label.addItems(["None"] + list(self.config.keys()))
        self.combo_label.setMaximumWidth(150)

        row2a.addWidget(self.btn_predict)
        row2a.addWidget(QtWidgets.QLabel("PF Label:"))
        row2a.addWidget(self.combo_label)

        self.main_layout.addWidget(QtWidgets.QLabel("Determine the PF of the NPj"))
        self.main_layout.addLayout(row2a)

        # =========================
        # ROW 2B — CALCULATE TIME
        # =========================
        row2b = QtWidgets.QHBoxLayout()

        self.btn_calculate = QtWidgets.QPushButton("Aggregate Time Estimation")
        self.btn_calculate.setFixedHeight(30)
        self.btn_calculate.setStyleSheet("font-size:15px")

        self.time_result_box = QtWidgets.QLineEdit()
        self.time_result_box.setReadOnly(True)
        self.time_result_box.setMaximumWidth(150)

        row2b.addWidget(self.btn_calculate)
        row2b.addWidget(QtWidgets.QLabel("Estimated Time:"))
        row2b.addWidget(self.time_result_box)
        self.time_result_box.setStyleSheet("font-size:13px; font-weight:bold;")

        self.main_layout.addWidget(QtWidgets.QLabel("Estimate Manufacturing Time"))
        self.main_layout.addLayout(row2b)

        # =========================
        # ROW 3 — RESULTS TABLE
        # =========================
        self.table_cases = QtWidgets.QTableWidget(0, 8)
        self.table_cases.setHorizontalHeaderLabels(
            ["Similarity", "Target Time", "Quantity", "Expanded Length", "Amorce", "Angle", "Thickness","Diameter"]
        )
        self.table_cases.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        cases_header = QtWidgets.QHBoxLayout()
        cases_header.addWidget(QtWidgets.QLabel("Top k:"))

        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setRange(1, 15)
        self.spin_k.setValue(5)
        self.spin_k.setFixedWidth(60)

        cases_header.addWidget(self.spin_k)
        cases_header.addSpacing(10)
        cases_header.addWidget(QtWidgets.QLabel("similar cases for explainability"))

        cases_header.addStretch()

        # Add header layout and table to main layout
        self.main_layout.addLayout(cases_header)
        self.main_layout.addWidget(self.table_cases)

        # ---------------- SIGNALS ----------------
        self.btn_process.clicked.connect(self.process_data)
        self.btn_predict.clicked.connect(self.predict_label)
        self.btn_calculate.clicked.connect(self.calculate_time)

    # ---------------- DATA ----------------
    def get_input_data(self):
        data = []
        for c in range(self.table_input.columnCount()):
            item = self.table_input.item(0, c)
            data.append(float(item.text()) if item and item.text() else -999)
        return data

    def process_data(self):
        for c in range(self.table_input.columnCount()):
            item = self.table_input.item(0, c)
            if item is None or item.text().strip() == "":
                self.table_input.setItem(
                    0, c, QtWidgets.QTableWidgetItem("-999")
                )

    # ---------------- PREDICT LABEL ----------------
    def predict_label(self):
        self.process_data()
        data = self.get_input_data()

        label = self.predictor.predict(data)
        print(f"Predicted label: {label}")

        if label not in self.config:
            label = "None"

        self.combo_label.setCurrentText(label)

    # ---------------- CBR ----------------
    def calculate_time(self):
        self.process_data()

        label = self.combo_label.currentText()
        if label == "None":
            return

        data = self.get_input_data()

        feature_order = self.config[label]["feature_order"]
        feature_map = {self.norm(f): i for i, f in enumerate(self.features)}

        values = []
        for key in feature_order:
            k = self.norm(key)
            if k in feature_map:
                values.append(data[feature_map[k]])

        predictor = self.kd_cbrs[label]

        cluster = predictor.predict_cluster(values)
        values.append(cluster)

        ok, result = predictor.predict_time(values, 5, self.spin_k.value())  # also respect spin_k

        if not ok:
            return

        self.time_result_box.setText(str(result["Predicted Value"]))

        top_n_rows = result["Top Rows"]
        self.table_cases.setRowCount(len(top_n_rows))

        # Column order: Similarity, Target Time, Quantity, Expanded Length, Amorce, Angle, Thickness, Diameter
        # Map display columns to their keys in df_sheet_80_dict
        col_keys = {
            "Similarity": None,  # special: from tuple[2]
            "Target Time": "Temps prévu",  # special: last key of the row dict
            "Quantity": "Qte",
            "Expanded Length": "developpé",
            "Amorce": "Amorce",
            "Angle": "angle",
            "Thickness": "eps",
            "Diameter": "diameter",
        }

        for i, (row_idx, local_sim, global_sim) in enumerate(top_n_rows):
            # Fetch the actual Excel row
            actual_row = predictor.df_sheet_80_dict[row_idx]
            last_key = list(actual_row.keys())[-1]  # Target Time column

            self.table_cases.setItem(i, 0, QtWidgets.QTableWidgetItem(str(global_sim)))
            self.table_cases.setItem(i, 1, QtWidgets.QTableWidgetItem(str(actual_row[last_key])))

            # Remaining columns — look up by key (case-insensitive fallback)
            display_cols = ["Quantity", "Expanded Length", "Amorce", "Angle", "Thickness", "Diameter"]
            keys = ["Qte", "developpé", "Amorce", "angle", "eps", "diameter"]

            for col_offset, key in enumerate(keys):
                # Try exact key first, then case-insensitive search
                value = actual_row.get(key)
                if value is None:
                    value = next(
                        (v for k, v in actual_row.items() if k.lower() == key.lower()),
                        "N/A"
                    )
                self.table_cases.setItem(
                    i, col_offset + 2,
                    QtWidgets.QTableWidgetItem(str(value))
                )


# ---------------- MAIN ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()