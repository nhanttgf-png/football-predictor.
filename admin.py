"""
admin.py
--------
Cấp / thu hồi Premium THỦ CÔNG — dùng cho hình thức thanh toán không tự
xác minh được (chuyển khoản ngân hàng qua mã QR tĩnh), khác với
payments.py (Stripe) vốn tự xác nhận qua webhook đã ký chữ ký.

Bảo vệ bằng MỘT mật khẩu admin riêng (biến môi trường ADMIN_TOKEN) —
KHÔNG phải mật khẩu đăng nhập của bất kỳ user nào. Tự đặt giá trị dài,
khó đoán, và KHÔNG bao giờ commit giá trị thật vào code / GitHub.
So sánh bằng hmac.compare_digest để tránh lộ thông tin qua thời gian xử
lý (timing attack).

Nếu chưa set ADMIN_TOKEN trên server, các endpoint này LUÔN từ chối —
tránh trường hợp quên cấu hình mà vô tình mở cửa cho bất kỳ ai.
"""

import os
import hmac
import logging
from datetime import datetime

from flask import Blueprint, request, jsonify, render_template

from extensions import db
from db_models import User

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__)

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _is_authorized() -> bool:
    if not ADMIN_TOKEN:
        return False
    supplied = request.headers.get("X-Admin-Token", "")
    return hmac.compare_digest(supplied, ADMIN_TOKEN)


@admin_bp.route("/admin")
def admin_page():
    """Trang HTML đơn giản để nhập mật khẩu admin + email khách rồi cấp
    Premium. Bản thân trang này không chứa dữ liệu nhạy cảm — bảo vệ
    thật sự nằm ở việc API bên dưới luôn đòi ADMIN_TOKEN đúng."""
    return render_template("admin.html")


@admin_bp.route("/api/admin/grant-premium", methods=["POST"])
def grant_premium():
    if not _is_authorized():
        return jsonify({"error": "Sai mật khẩu admin (hoặc server chưa cấu hình ADMIN_TOKEN)."}), 401

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Thiếu email."}), 400

    user = User.query.filter_by(email=email).first()
    if user is None:
        return jsonify({"error": f"Không tìm thấy tài khoản với email {email}."}), 404

    user.is_premium = True
    user.premium_since = datetime.utcnow()
    db.session.commit()
    logger.info("Admin đã cấp Premium thủ công cho %s", email)
    return jsonify({"message": f"Đã cấp Premium cho {email}.", "email": email, "premium": True})


@admin_bp.route("/api/admin/revoke-premium", methods=["POST"])
def revoke_premium():
    if not _is_authorized():
        return jsonify({"error": "Sai mật khẩu admin (hoặc server chưa cấu hình ADMIN_TOKEN)."}), 401

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Thiếu email."}), 400

    user = User.query.filter_by(email=email).first()
    if user is None:
        return jsonify({"error": f"Không tìm thấy tài khoản với email {email}."}), 404

    user.is_premium = False
    db.session.commit()
    logger.info("Admin đã thu hồi Premium của %s", email)
    return jsonify({"message": f"Đã thu hồi Premium của {email}.", "email": email, "premium": False})
