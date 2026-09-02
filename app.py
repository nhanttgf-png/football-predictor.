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
import uuid
from datetime import date
from urllib.parse import quote

from flask import Flask, render_template, request, jsonify, session
from flask_login import current_user, login_required

from extensions import db, login_manager
from db_models import User
from auth import auth_bp
from payments import payments_bp
from admin import admin_bp
from model import load_or_train_model, predict_match

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Cần secret key để ký cookie session (đăng nhập + định danh khách ẩn danh).
# Ở production PHẢI đặt biến môi trường SECRET_KEY, không dùng giá trị
# mặc định này (ai biết giá trị mặc định có thể giả mạo cookie đăng nhập).
app.secret_key = os.environ.get("SECRET_KEY", "doi-secret-key-nay-khi-deploy-that")

# ==================== DATABASE (tài khoản, trạng thái Premium) ====================
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///" + os.path.join(basedir, "app.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
login_manager.init_app(app)
# API trả JSON, không phải trang HTML có form đăng nhập -> khi chưa đăng
# nhập mà gọi route @login_required, không redirect (mặc định của
# Flask-Login) mà chỉ cần trả 401 cho FE tự xử lý.
login_manager.login_view = None


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Vui lòng đăng nhập trước."}), 401


with app.app_context():
    db.create_all()

app.register_blueprint(auth_bp)
app.register_blueprint(payments_bp)
app.register_blueprint(admin_bp)

# ==================== NÂNG CẤP PREMIUM QUA CHUYỂN KHOẢN (QR TĨNH) ====
# Cách này KHÔNG tự xác nhận thanh toán (khác Stripe ở payments.py) — sau
# khi khách chuyển khoản, ADMIN tự kiểm tra sao kê ngân hàng rồi cấp
# Premium thủ công qua trang /admin (xem admin.py). Cấu hình 3 biến môi
# trường dưới đây trên Render (Settings > Environment):
#   BANK_ID            Mã ngân hàng theo chuẩn VietQR, vd "MB", "VCB",
#                       "TCB"... (tra đầy đủ tại https://api.vietqr.io/v2/banks)
#   BANK_ACCOUNT_NO     Số tài khoản nhận tiền
#   BANK_ACCOUNT_NAME   Tên chủ tài khoản (không dấu, đúng như trên tài khoản)
# Giá gói có thể đổi qua biến PREMIUM_PRICE_VND (mặc định 19000).
BANK_ID = os.environ.get("BANK_ID", "")
BANK_ACCOUNT_NO = os.environ.get("BANK_ACCOUNT_NO", "")
BANK_ACCOUNT_NAME = os.environ.get("BANK_ACCOUNT_NAME", "")
PREMIUM_PRICE_VND = int(os.environ.get("PREMIUM_PRICE_VND", "19000"))


@app.route("/api/premium-qr")
@login_required
def api_premium_qr():
    """Trả về thông tin QR chuyển khoản (dùng ảnh QR động của VietQR.io,
    không cần API key) để FE hiển thị trong modal nâng cấp Premium.
    Nội dung chuyển khoản gắn ID của user để admin đối chiếu khi duyệt."""
    if not BANK_ID or not BANK_ACCOUNT_NO:
        return jsonify({
            "error": "Server chưa cấu hình tài khoản ngân hàng nhận Premium "
                     "(thiếu BANK_ID / BANK_ACCOUNT_NO)."
        }), 500

    transfer_content = f"PREMIUM {current_user.id}"
    qr_url = (
        f"https://img.vietqr.io/image/{BANK_ID}-{BANK_ACCOUNT_NO}-compact2.png"
        f"?amount={PREMIUM_PRICE_VND}"
        f"&addInfo={quote(transfer_content)}"
        f"&accountName={quote(BANK_ACCOUNT_NAME)}"
    )
    return jsonify({
        "qr_url": qr_url,
        "bank_id": BANK_ID,
        "account_no": BANK_ACCOUNT_NO,
        "account_name": BANK_ACCOUNT_NAME,
        "amount": PREMIUM_PRICE_VND,
        "transfer_content": transfer_content,
    })

# ==================== GIỚI HẠN LƯỢT DÙNG MIỄN PHÍ (KHÁCH CHƯA ĐĂNG NHẬP) ====
# Với khách chưa đăng nhập, vẫn cho dùng thử 3 lượt/ngày qua cookie session
# ẩn danh (như bản trước) — không ép phải đăng ký mới xem được demo. Nhưng
# lưu ý: đây là bộ nhớ trong tiến trình (in-memory), không sống sót qua
# restart / không đồng bộ giữa nhiều worker. Người dùng ĐÃ đăng nhập không
# dùng cơ chế này nữa — sẽ dùng cột free_predictions_used trong DB
# (db_models.User), bền và đúng khi chạy nhiều worker.
FREE_DAILY_LIMIT = 3
_guest_usage_store = {}  # { session_uid: {"date": "YYYY-MM-DD", "count": int} }


def _get_guest_id() -> str:
    if "uid" not in session:
        session["uid"] = uuid.uuid4().hex
        session.permanent = True
    return session["uid"]


def _get_guest_usage() -> dict:
    today = date.today().isoformat()
    uid = _get_guest_id()
    rec = _guest_usage_store.get(uid)
    if rec is None or rec["date"] != today:
        rec = {"date": today, "count": 0}
        _guest_usage_store[uid] = rec
    return rec


def _guest_usage_payload(rec: dict) -> dict:
    return {
        "limit": FREE_DAILY_LIMIT,
        "used": rec["count"],
        "remaining": max(0, FREE_DAILY_LIMIT - rec["count"]),
        "premium": False,
    }

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


@app.route("/api/usage")
def api_usage():
    """Số lượt dự đoán miễn phí còn lại trong ngày + trạng thái Premium.
    Ưu tiên dữ liệu tài khoản (DB) nếu đã đăng nhập, không thì dùng cookie
    ẩn danh cho khách."""
    if current_user.is_authenticated:
        return jsonify(current_user.usage_dict(FREE_DAILY_LIMIT))
    return jsonify(_guest_usage_payload(_get_guest_usage()))


@app.route("/api/upgrade", methods=["POST"])
def api_upgrade():
    """
    CHỈ dùng để TEST LOCAL khi bạn chưa cấu hình Stripe (STRIPE_SECRET_KEY
    trống) — set biến môi trường ENABLE_DEMO_UPGRADE=1 để bật. Mặc định bị
    tắt vì đây KHÔNG phải xác nhận thanh toán thật; nếu bật ở production,
    bất kỳ ai gọi API này cũng tự cấp Premium miễn phí cho chính họ.
    Thanh toán thật đi qua /api/create-checkout-session (payments.py) và
    chỉ được xác nhận qua webhook đã ký của Stripe (/api/stripe-webhook).
    """
    if os.environ.get("ENABLE_DEMO_UPGRADE") != "1":
        return jsonify({
            "error": "Endpoint demo này đang tắt. Dùng nút Nâng cấp Premium "
                     "thật (yêu cầu đăng nhập) để thanh toán qua Stripe."
        }), 403
    if not current_user.is_authenticated:
        return jsonify({"error": "Vui lòng đăng nhập trước."}), 401

    current_user.is_premium = True
    db.session.commit()
    return jsonify({"message": "Đã nâng cấp Premium (demo, chỉ để test local).",
                     "usage": current_user.usage_dict(FREE_DAILY_LIMIT)})


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

    is_logged_in = current_user.is_authenticated
    guest_rec = None if is_logged_in else _get_guest_usage()

    can_predict = current_user.can_predict(FREE_DAILY_LIMIT) if is_logged_in else guest_rec["count"] < FREE_DAILY_LIMIT
    is_premium = current_user.is_premium if is_logged_in else False

    if not can_predict:
        usage = current_user.usage_dict(FREE_DAILY_LIMIT) if is_logged_in else _guest_usage_payload(guest_rec)
        return jsonify({
            "error": f"Bạn đã dùng hết {FREE_DAILY_LIMIT} lượt dự đoán miễn phí hôm nay. "
                     "Nâng cấp Premium để dự đoán không giới hạn và xem thống kê chi tiết.",
            "limit_reached": True,
            "usage": usage,
        }), 402

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

    if is_logged_in:
        current_user.register_prediction()
        db.session.commit()
        usage = current_user.usage_dict(FREE_DAILY_LIMIT)
    else:
        guest_rec["count"] += 1
        usage = _guest_usage_payload(guest_rec)

    if not is_premium:
        result["premium_stats"] = None  # chặn thống kê chi tiết với tài khoản/khách free

    result["usage"] = usage
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
