"""
app.py
------
Flask app: vừa phục vụ giao diện web (templates/index.html),
vừa cung cấp API cho phần AI dự đoán tỉ lệ bóng đá.

Chạy thử ở local: python app.py
Mặc định chạy tại http://127.0.0.1:5000

Khi deploy (Render/Railway...), server sẽ chạy bằng gunicorn:
    gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os

from flask import Flask, render_template, request, jsonify

from model import load_or_train_model, predict_match

app = Flask(__name__)

# Không huấn luyện ngay khi import module nữa, vì trên các nền tảng deploy
# (Render, Railway...) việc tải dữ liệu FBref có thể mất vài chục giây,
# dễ khiến server bị coi là "khởi động thất bại" (timeout).
# Thay vào đó, mô hình sẽ được huấn luyện/khôi phục ở LẦN GỌI ĐẦU TIÊN.
_state = {"model": None, "home_stats": None, "away_stats": None, "teams": None}


def get_model():
    if _state["model"] is None:
        print("Đang tải mô hình...")
        try:
            model, home_stats, away_stats, teams = load_or_train_model()
        except Exception as e:
            # Trên server deploy (không có Chrome), việc tự tải dữ liệu mới sẽ
            # lỗi nếu chưa có sẵn model_cache.pkl. Xem hướng dẫn train_offline.py.
            raise RuntimeError(
                "Chưa có model_cache.pkl và server này không tự tải dữ liệu được "
                "(thiếu Chrome). Hãy chạy `python train_offline.py` ở máy local rồi "
                "commit + push file model_cache.pkl lên GitHub."
            ) from e
        _state.update(
            model=model, home_stats=home_stats, away_stats=away_stats, teams=teams
        )
        print(f"Xong! Đã có dữ liệu của {len(teams)} đội.")
    return _state["model"], _state["home_stats"], _state["away_stats"], _state["teams"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    """Trả về danh sách các đội để frontend hiển thị dropdown."""
    try:
        _, _, _, teams = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
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
        model, home_stats, away_stats, _ = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        result = predict_match(model, home_stats, away_stats, doi_nha, doi_khach)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Huấn luyện lại mô hình từ đầu (tải dữ liệu mới nhất từ FBref)."""
    model, home_stats, away_stats, teams = load_or_train_model(force_retrain=True)
    _state.update(
        model=model, home_stats=home_stats, away_stats=away_stats, teams=teams
    )
    return jsonify({"message": "Đã huấn luyện lại mô hình.", "so_doi": len(teams)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)