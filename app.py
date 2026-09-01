"""
app.py
------
Flask app: vừa phục vụ giao diện web (templates/index.html),
vừa cung cấp API cho phần AI dự đoán tỉ lệ bóng đá.

Chạy thử: python app.py
Mặc định chạy tại http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify

from model import load_or_train_model, predict_match

app = Flask(__name__)

# Huấn luyện (hoặc load cache) mô hình MỘT LẦN khi server khởi động,
# để mỗi request /api/predict không phải tải lại dữ liệu từ FBref.
print("Đang tải dữ liệu và huấn luyện mô hình, vui lòng chờ...")
model, home_stats, away_stats, teams = load_or_train_model()
print(f"Xong! Đã có dữ liệu của {len(teams)} đội.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    """Trả về danh sách các đội để frontend hiển thị dropdown."""
    return jsonify({"teams": teams})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Body JSON mong đợi: { "doi_nha": "Arsenal", "doi_khach": "Chelsea" }
    """
    data = request.get_json(silent=True) or {}
    doi_nha = data.get("doi_nha", "").strip()
    doi_khach = data.get("doi_khach", "").strip()

    if not doi_nha or not doi_khach:
        return jsonify({"error": "Vui lòng chọn cả đội nhà và đội khách."}), 400

    if doi_nha == doi_khach:
        return jsonify({"error": "Hai đội phải khác nhau."}), 400

    try:
        result = predict_match(model, home_stats, away_stats, doi_nha, doi_khach)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Huấn luyện lại mô hình từ đầu (tải dữ liệu mới nhất từ FBref)."""
    global model, home_stats, away_stats, teams
    model, home_stats, away_stats, teams = load_or_train_model(force_retrain=True)
    return jsonify({"message": "Đã huấn luyện lại mô hình.", "so_doi": len(teams)})


if __name__ == "__main__":
    app.run(debug=True)
