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
from db_models import User, ChallengeMatch, ChallengeGuess

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


# ==================== THỬ THÁCH DỰ ĐOÁN (tạo trận / chốt kết quả) ====================
#
# Import get_model/_valid_league_key từ app.py NGAY TRONG THÂN HÀM (không
# import ở đầu file) để tránh vòng lặp import: app.py import admin_bp từ
# module này trước khi get_model được định nghĩa xong; tới lúc các route
# này thực sự được GỌI (mỗi request) thì app.py đã import xong hoàn toàn
# nên import ở đây luôn an toàn.

@admin_bp.route("/api/admin/challenge/create", methods=["POST"])
def challenge_create():
    """Tạo 1 trận thử thách mới: admin chỉ cần chọn giải + 2 đội, AI tự
    tính tỷ lệ thắng/hòa/thua ngay lúc tạo (đóng băng lại để so sánh công
    bằng với lượt đoán của người dùng, không đổi dù model có retrain sau).
    Body JSON: { "league": "premier-league", "doi_nha": "Arsenal",
                 "doi_khach": "Chelsea", "close_others": false } """
    if not _is_authorized():
        return jsonify({"error": "Sai mật khẩu admin (hoặc server chưa cấu hình ADMIN_TOKEN)."}), 401

    from app import get_model, _valid_league_key, DEFAULT_LEAGUE_KEY
    from model import predict_match

    data = request.get_json(silent=True) or {}
    doi_nha = (data.get("doi_nha") or "").strip()
    doi_khach = (data.get("doi_khach") or "").strip()
    league_key = _valid_league_key(data.get("league", DEFAULT_LEAGUE_KEY))

    if not doi_nha or not doi_khach:
        return jsonify({"error": "Vui lòng nhập cả đội nhà và đội khách."}), 400
    if doi_nha == doi_khach:
        return jsonify({"error": "Hai đội phải khác nhau."}), 400

    try:
        model, team_state, _, _ = get_model(league_key)
        result = predict_match(model, team_state, doi_nha, doi_khach)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Lỗi khi tạo trận thử thách: %s", e, exc_info=True)
        return jsonify({"error": "Có lỗi xảy ra khi tính dự đoán AI cho trận này."}), 500

    probs = {"nha": result["thang_nha"], "hoa": result["hoa"], "khach": result["thang_khach"]}
    ai_du_doan = max(probs, key=probs.get)

    if data.get("close_others"):
        for m in ChallengeMatch.query.filter_by(is_active=True, ket_qua_thuc_te=None).all():
            m.is_active = False

    match = ChallengeMatch(
        league=league_key,
        doi_nha=doi_nha,
        doi_khach=doi_khach,
        ai_thang_nha=result["thang_nha"],
        ai_hoa=result["hoa"],
        ai_thang_khach=result["thang_khach"],
        ai_du_doan=ai_du_doan,
    )
    db.session.add(match)
    db.session.commit()
    logger.info("Admin đã tạo trận thử thách #%d: %s vs %s (%s)", match.id, doi_nha, doi_khach, league_key)

    return jsonify({"message": "Đã tạo trận thử thách mới.", "match": match.to_dict()}), 201


@admin_bp.route("/api/admin/challenge/settle", methods=["POST"])
def challenge_settle():
    """Chốt sổ 1 trận thử thách với kết quả thật ngoài đời, tự cộng XP cho
    mọi lượt đoán đúng. Body JSON: { "match_id": 1, "ket_qua_thuc_te":
    "nha" | "hoa" | "khach", "ty_so_thuc_te": "2-1" (không bắt buộc) }"""
    if not _is_authorized():
        return jsonify({"error": "Sai mật khẩu admin (hoặc server chưa cấu hình ADMIN_TOKEN)."}), 401

    from app import CHALLENGE_POINTS

    data = request.get_json(silent=True) or {}
    match_id = data.get("match_id")
    ket_qua = (data.get("ket_qua_thuc_te") or "").strip()
    ty_so = (data.get("ty_so_thuc_te") or "").strip() or None

    if ket_qua not in ("nha", "hoa", "khach"):
        return jsonify({"error": "ket_qua_thuc_te phải là nha, hoa hoặc khach."}), 400

    match = ChallengeMatch.query.get(match_id)
    if match is None:
        return jsonify({"error": "Không tìm thấy trận thử thách này."}), 404
    if match.ket_qua_thuc_te is not None:
        return jsonify({"error": "Trận này đã được chốt sổ rồi."}), 409

    match.ket_qua_thuc_te = ket_qua
    match.ty_so_thuc_te = ty_so
    match.is_active = False
    match.settled_at = datetime.utcnow()

    so_dung = 0
    tong_diem = 0
    for guess in match.guesses:
        diem = CHALLENGE_POINTS[ket_qua] if guess.du_doan == ket_qua else 0
        guess.diem = diem
        if diem:
            so_dung += 1
            tong_diem += diem
            user = User.query.get(guess.user_id)
            if user is not None:
                user.total_xp = (user.total_xp or 0) + diem

    db.session.commit()
    logger.info(
        "Admin đã chốt sổ trận thử thách #%d: kết quả=%s, %d/%d lượt đoán đúng, +%d XP tổng cộng",
        match.id, ket_qua, so_dung, match.guesses.count(), tong_diem,
    )

    return jsonify({
        "message": "Đã chốt sổ trận thử thách và cộng XP.",
        "match": match.to_dict(),
        "so_luot_doan": match.guesses.count(),
        "so_dung": so_dung,
        "tong_diem_da_cong": tong_diem,
    })


@admin_bp.route("/api/admin/challenge/list")
def challenge_list():
    """Danh sách các trận thử thách gần đây (cả đang mở lẫn đã chốt sổ),
    để admin theo dõi và biết trận nào cần chốt sổ."""
    if not _is_authorized():
        return jsonify({"error": "Sai mật khẩu admin (hoặc server chưa cấu hình ADMIN_TOKEN)."}), 401

    rows = ChallengeMatch.query.order_by(ChallengeMatch.created_at.desc()).limit(30).all()
    return jsonify({"rows": [m.to_dict() for m in rows]})
