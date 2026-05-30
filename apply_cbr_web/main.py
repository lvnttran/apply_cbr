from pathlib import Path

from flask import Flask, jsonify, render_template, request
import yaml

from utils.kd_cbr_base import KdCbrBase
from utils.predictor import Predictor
from dto import EstimateResponseDTO, PredictResponseDTO

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"

app = Flask(__name__)


def make_abs_path(path_value):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for label, entry in config.items():
        for key in ["excel_file", "kmeans_model_path", "train_excel_path", "test_excel_path"]:
            if key in entry:
                entry[key] = make_abs_path(entry[key])
    return config


config = load_config()
predictor = Predictor()
kd_cbrs = {label: KdCbrBase(cfg) for label, cfg in config.items()}

DEFAULT_FEATURES = [
    "Angle",
    "Long",
    "Long larg",
    "Diameter",
    "Eps",
    "Hauteur",
    "Amorce",
    "Dimension",
    "Developpé",
    "Qte",
    "Diam circle",
    "Larg",
]


@app.route("/")
def index():
    return render_template("index.html", features=DEFAULT_FEATURES, labels=list(config.keys()))


@app.route("/api/labels", methods=["GET"])
def get_labels():
    return jsonify(sorted(list(config.keys())))


def normalize_key(value):
    return str(value).replace("_", " ").replace("é", "e").strip().lower()


def parse_features(data):
    if isinstance(data, dict) and "features" in data:
        return [float(x) for x in data["features"]]

    if isinstance(data, list):
        return [float(x) for x in data]

    if isinstance(data, dict):
        values = []
        for name in DEFAULT_FEATURES:
            if name in data:
                values.append(float(data[name]))
            elif normalize_key(name) in data:
                values.append(float(data[normalize_key(name)]))
            else:
                raise ValueError(f"Missing feature: {name}")
        return values

    raise ValueError("Invalid feature payload")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    features = parse_features(payload)
    label = predictor.predict(features)
    return jsonify(PredictResponseDTO(label=label).to_dict())


@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    payload = request.get_json(force=True)
    features = parse_features(payload)
    label = payload.get("label")

    if label is None:
        label = predictor.predict(features)

    if label not in config:
        return jsonify({"error": f"Unknown label: {label}"}), 400

    feature_order = config[label]["feature_order"]
    feature_map = {normalize_key(name): features[i] for i, name in enumerate(DEFAULT_FEATURES)}

    values = []
    for key in feature_order:
        normalized_key = normalize_key(key)
        if normalized_key not in feature_map:
            return jsonify({"error": f"Missing feature for label '{label}': {key}"}), 400
        values.append(feature_map[normalized_key])

    predictor_model = kd_cbrs[label]
    cluster = predictor_model.predict_cluster(values)
    values.append(cluster)

    ok, result = predictor_model.predict_time(values, k=5, n=5)
    if not ok:
        return jsonify({"error": result}), 500

    # ADD THIS LINE
    top_rows_data = [predictor_model.df_sheet_80_dict[row[0]] for row in result["Top Rows"]]

    response_dto = EstimateResponseDTO(
        label=label,
        cluster=cluster,
        predicted_value=result["Predicted Value"],
        top_rows=result["Top Rows"],
        top_rows_data=top_rows_data,
        updated_input=result["Updated New Value"],
    )
    return jsonify(response_dto.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
