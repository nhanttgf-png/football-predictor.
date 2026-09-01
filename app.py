"""
app.py
------
Flask web app cho dự đoán bóng đá.
Sử dụng model đã train (từ cache hoặc train online).

LƯU Ý (sửa 2 lỗi so với bản trước):
1. `@app.before_first_request` đã bị GỠ BỎ khỏi Flask từ bản 2.3 trở đi.
   Vì requirements.txt không ghim version Flask, cài đặt sẽ luôn ra bản mới
   nhất -> code cũ dùng decorator này sẽ crash ngay khi import (AttributeError).
   Ở đây quay lại cách lazy-load quen thuộc: chỉ init model ở LẦN GỌI ĐẦU TIÊN
   thông qua hàm get_model(), không phụ thuộc vào decorator nào của Flask.
2. Frontend (static/script.js) gọi các endpoint /api/teams, /api/predict,
   /api/model-info -- nhưng bản trước chỉ có /predict (thiếu tiền tố /api),
   không có /api/teams, /api/model-info -> web luôn báo lỗi 404. Đã khôi
   phục đúng các đường dẫn này.
"""

import os
import logging

from flask import Flask, render_template, request, jsonify

from model import load_or_train_model, predict_match

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Không train ngay khi import module, vì trên các nền tảng deploy (Render,
# Railway...) việc tải dữ liệu FBref có thể mất khá lâu, dễ khiến server bị
# coi là "khởi động thất bại" (timeout). Thay vào đó, model được
# train/khôi phục ở LẦN GỌI ĐẦU TIÊN tới get_model().
_state = {"model": None, "team_state": None, "teams": None, "metrics": None}


def get_model():
    if _state["model"] is None:
        logger.info("Đang tải mô hình...")
        try:
            model, team_state, teams, metrics = load_or_train_model()
        except Exception as e:
            logger.error("Lỗi khởi tạo model: %s", e, exc_info=True)
            raise RuntimeError(
                "Chưa có model_cache.pkl và server này không tự tải dữ liệu được "
                "(thiếu Chrome). Hãy chạy `python train_offline.py` ở máy local rồi "
                "commit + push file model_cache.pkl lên GitHub."
            ) from e
        _state.update(model=model, team_state=team_state, teams=teams, metrics=metrics)
        logger.info("Model đã sẵn sàng! Số đội: %d", len(teams))
        if metrics.get("accuracy") is not None:
            logger.info(
                "Metrics: accuracy=%s%%, log_loss=%s, model_type=%s",
                metrics.get("accuracy"), metrics.get("log_loss"), metrics.get("model_type"),
            )
    return _state["model"], _state["team_state"], _state["teams"], _state["metrics"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    """Trả về danh sách các đội để frontend hiển thị dropdown."""
    try:
        _, _, teams, _ = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"teams": teams})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Body JSON mong đợi: { "doi_nha": "Arsenal", "doi_khach": "Chelsea" }
    """
    data = request.get_json(silent=True) or {}
    doi_nha = (data.get("doi_nha") or "").strip()
    doi_khach = (data.get("doi_khach") or "").strip()

    if not doi_nha or not doi_khach:
        return jsonify({"error": "Vui lòng chọn cả đội nhà và đội khách."}), 400

    if doi_nha == doi_khach:
        return jsonify({"error": "Hai đội phải khác nhau."}), 400

    try:
        model, team_state, _, _ = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        result = predict_match(model, team_state, doi_nha, doi_khach)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Lỗi khi dự đoán: %s", e, exc_info=True)
        return jsonify({"error": "Có lỗi xảy ra, vui lòng thử lại."}), 500

    return jsonify(result)


@app.route("/api/model-info")
def api_model_info():
    """
    Trả về độ chính xác thật của model, đo trên tập test tách theo thời gian
    (không phải số liệu "ảo" từ chính dữ liệu train). Frontend dùng để hiển
    thị cho người dùng biết nên tin dự đoán tới mức nào.
    """
    try:
        _, _, _, metrics = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(metrics)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Huấn luyện lại mô hình từ đầu (tải dữ liệu mới nhất từ FBref)."""
    try:
        model, team_state, teams, metrics = load_or_train_model(force_retrain=True)
    except Exception as e:
        logger.error("Lỗi khi retrain: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    _state.update(model=model, team_state=team_state, teams=teams, metrics=metrics)
    return jsonify({
        "message": "Đã huấn luyện lại mô hình.",
        "so_doi": len(teams),
        "metrics": metrics,
    })


@app.route("/health")
def health():
    """Health check endpoint, hữu ích khi deploy (Render/Railway)."""
    return jsonify({
        "status": "healthy",
        "model_ready": _state["model"] is not None,
        "num_teams": len(_state["teams"]) if _state["teams"] else 0,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
