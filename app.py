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
from db_models import User, Prediction, ChallengeMatch, ChallengeGuess
from auth import auth_bp
from payments import payments_bp
from admin import admin_bp
from model import (
    load_or_train_model,
    predict_match,
    LEAGUES,
    DEFAULT_LEAGUE_KEY,
    build_leaderboard,
    sort_leaderboard,
    LEADERBOARD_SORT_FIELDS,
)
from player_ratings import (
    load_or_build_player_ratings,
    build_player_leaderboard,
    sort_player_leaderboard,
    get_key_players,
    PLAYER_SORT_FIELDS,
    POSITIONS as PLAYER_POSITIONS,
)

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


def _run_startup_migrations():
    """
    Tự thêm các CỘT MỚI vào bảng đã tồn tại từ trước, mỗi khi app khởi động.

    Lý do cần cái này: db.create_all() (gọi ngay dưới) chỉ tạo BẢNG MỚI
    (vd challenge_matches, challenge_guesses) -- nó KHÔNG tự thêm cột mới
    vào 1 bảng đã tồn tại sẵn (vd thêm users.total_xp vào bảng "users" đã
    có dữ liệu) -> nếu không có hàm này, app sẽ crash với lỗi kiểu
    "no such column: users.total_xp" ngay khi vừa deploy code mới lên 1
    server đã có app.db từ trước.

    Chạy tự động ở MỌI lần khởi động (kể cả trên Render sau mỗi lần push
    GitHub) nên bạn không cần SSH vào server để chạy migration thủ công.
    An toàn để chạy lại nhiều lần: bỏ qua ngay nếu cột đã tồn tại rồi.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "users" not in inspector.get_table_names():
        return  # bảng users chưa từng tồn tại -> create_all() ở trên đã tạo đủ cột

    existing_cols = {c["name"] for c in inspector.get_columns("users")}
    if "total_xp" not in existing_cols:
        logger.info("Đang tự thêm cột users.total_xp còn thiếu (migration)...")
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN total_xp INTEGER NOT NULL DEFAULT 0"))
        logger.info("Đã thêm cột users.total_xp.")


with app.app_context():
    db.create_all()
    _run_startup_migrations()

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

# ---- Điểm XP cho tính năng "Thử thách dự đoán" (Prediction Challenge) ----
# Đoán đúng kèo hòa khó hơn (xác suất thấp hơn) nên được thưởng cao hơn.
CHALLENGE_POINTS = {"nha": 10, "khach": 10, "hoa": 15}


def _mask_email(email: str) -> str:
    """Che bớt email để hiển thị công khai trên bảng xếp hạng XP,
    vd "nguyen@gmail.com" -> "ng***@gmail.com"."""
    local, _, domain = (email or "").partition("@")
    if not local:
        return "ẩn danh"
    visible = local[:2]
    return f"{visible}***@{domain}" if domain else f"{visible}***"


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
#
# _state giờ là dict-of-dict, 1 entry riêng cho mỗi giải đấu, vì mỗi giải
# có model/team_state/teams/metrics hoàn toàn khác nhau.
_state = {}


def _empty_state():
    return {"model": None, "team_state": None, "teams": None, "metrics": None}


def _valid_league_key(league_key: str) -> str:
    """Trả về league_key hợp lệ, hoặc mặc định nếu người dùng gửi giá trị lạ."""
    return league_key if league_key in LEAGUES else DEFAULT_LEAGUE_KEY


# ==================== RATING CẦU THỦ ====================
# Cùng lý do với model dự đoán trận đấu: KHÔNG tự tải dữ liệu FBref lúc
# import module, chỉ tải/khôi phục cache ở lần gọi đầu tiên.
_player_state = {}


def _empty_player_state():
    return {"players": None, "meta": None}


def _default_player_season(league_key: str) -> str:
    """Rating cầu thủ dùng số liệu 1 MÙA GIẢI cụ thể (khác model dự đoán
    trận đấu, gộp nhiều mùa). Mặc định lấy mùa GẦN NHẤT ĐÃ ĐÁ ĐỦ (mùa
    liền trước mùa hiện tại), vì đầu mùa mới cầu thủ chưa đá đủ số phút
    tối thiểu (MIN_MINUTES) để rating có ý nghĩa."""
    seasons = LEAGUES[league_key]["seasons"]
    return seasons[-2] if len(seasons) >= 2 else seasons[-1]


def get_player_ratings(league_key: str = DEFAULT_LEAGUE_KEY, season: str = None):
    league_key = _valid_league_key(league_key)
    season = season or _default_player_season(league_key)
    cache_key = f"{league_key}:{season}"
    state = _player_state.setdefault(cache_key, _empty_player_state())

    if state["players"] is None:
        logger.info("Đang tải rating cầu thủ cho giải %s, mùa %s...", league_key, season)
        try:
            players, meta = load_or_build_player_ratings(
                league_key=league_key,
                league_code=LEAGUES[league_key]["code"],
                season=season,
            )
        except Exception as e:
            logger.error("Lỗi khởi tạo rating cầu thủ (%s): %s", league_key, e, exc_info=True)
            raise RuntimeError(
                f"Chưa có cache rating cầu thủ cho giải \"{LEAGUES[league_key]['name']}\" "
                f"(mùa {season}) và server này không tự tải dữ liệu được (thiếu Chrome). "
                f"Hãy chạy `python train_players_offline.py --league "
                f"\"{LEAGUES[league_key]['code']}\" --season {season}` ở máy local rồi "
                "commit + push file player_ratings_<giải>.pkl tương ứng lên GitHub."
            ) from e
        state.update(players=players, meta=meta)
        logger.info("Rating cầu thủ %s đã sẵn sàng! Số cầu thủ: %d", league_key, len(players))
    return state["players"], state["meta"]


# ==================== GOOGLE ADSENSE ====================
# Chỉ hiện quảng cáo khi ADSENSE_CLIENT được cấu hình (mã dạng ca-pub-...,
# lấy trong AdSense Dashboard > Sites, sau khi được Google DUYỆT site).
# Chưa duyệt / chưa set biến này -> toàn bộ script + khung quảng cáo
# KHÔNG được render, trang chạy y như hiện tại, không có gì thay đổi.
ADSENSE_CLIENT = os.environ.get("ADSENSE_CLIENT", "")
ADSENSE_SLOT_TOP = os.environ.get("ADSENSE_SLOT_TOP", "")
ADSENSE_SLOT_MID = os.environ.get("ADSENSE_SLOT_MID", "")
ADSENSE_SLOT_FOOTER = os.environ.get("ADSENSE_SLOT_FOOTER", "")


@app.route("/ads.txt")
def ads_txt():
    """Google AdSense yêu cầu file này ở gốc domain để xác minh quyền sở
    hữu quảng cáo, tránh bị giả mạo site bán quảng cáo hộ bạn."""
    if not ADSENSE_CLIENT:
        return "", 404
    pub_id = ADSENSE_CLIENT.replace("ca-pub-", "pub-")
    return f"google.com, {pub_id}, DIRECT, f08c47fec0942fa0\n", 200, {"Content-Type": "text/plain"}


@app.route("/privacy")
def privacy_page():
    return render_template("privacy.html")


@app.route("/terms")
def terms_page():
    return render_template("terms.html")


@app.route("/")
def index():
    return render_template(
        "index.html",
        adsense_client=ADSENSE_CLIENT,
        adsense_slot_top=ADSENSE_SLOT_TOP,
        adsense_slot_mid=ADSENSE_SLOT_MID,
        adsense_slot_footer=ADSENSE_SLOT_FOOTER,
    )


def get_model(league_key: str = DEFAULT_LEAGUE_KEY):
    league_key = _valid_league_key(league_key)
    state = _state.setdefault(league_key, _empty_state())

    if state["model"] is None:
        logger.info("Đang tải mô hình cho giải %s...", league_key)
        try:
            model, team_state, teams, metrics = load_or_train_model(league_key=league_key)
        except Exception as e:
            logger.error("Lỗi khởi tạo model (%s): %s", league_key, e, exc_info=True)
            raise RuntimeError(
                f"Chưa có model cache cho giải \"{LEAGUES[league_key]['name']}\" và "
                "server này không tự tải dữ liệu được (thiếu Chrome). Hãy chạy "
                f"`python train_offline.py --league \"{LEAGUES[league_key]['code']}\"` "
                "ở máy local rồi commit + push file model_cache tương ứng lên GitHub."
            ) from e
        state.update(model=model, team_state=team_state, teams=teams, metrics=metrics)
        logger.info("Model %s đã sẵn sàng! Số đội: %d", league_key, len(teams))
        if metrics.get("accuracy") is not None:
            logger.info(
                "Metrics (%s): accuracy=%s%%, log_loss=%s, model_type=%s",
                league_key, metrics.get("accuracy"), metrics.get("log_loss"), metrics.get("model_type"),
            )
    return state["model"], state["team_state"], state["teams"], state["metrics"]


def _backfill_predictions(league_key: str, team_state: dict) -> None:
    """Đối chiếu các dự đoán đang chờ kết quả (ket_qua_thuc_te is None) của
    giải `league_key` với dữ liệu đối đầu (h2h_hist) mới nhất trong
    team_state. h2h_hist lưu (đội_nhà_lúc_đó, hiệu_số_bàn, ngày) cho mọi
    cặp đội đã từng gặp nhau -- nếu tìm thấy 1 trận diễn ra SAU thời điểm
    dự đoán được tạo, coi đó là kết quả thật của trận đã dự đoán.

    Đây là cách xấp xỉ hợp lý mà KHÔNG cần thêm 1 nguồn dữ liệu lịch thi
    đấu/kết quả trực tiếp riêng: kết quả chỉ "về" sau khi model được
    retrain với dữ liệu mới có chứa trận đấu đó.
    """
    pending = Prediction.query.filter_by(league=league_key, ket_qua_thuc_te=None).all()
    if not pending:
        return
    h2h_hist = team_state.get("h2h_hist", {})
    changed = False
    for pred in pending:
        pair_key = frozenset((pred.doi_nha, pred.doi_khach))
        for past_home, diff, match_date in h2h_hist.get(pair_key, []):
            if match_date is None or match_date.replace(tzinfo=None) < pred.created_at:
                continue
            oriented = diff if past_home == pred.doi_nha else -diff
            ket_qua = "hoa" if oriented == 0 else ("nha" if oriented > 0 else "khach")
            pred.ket_qua_thuc_te = ket_qua
            pred.dung = (ket_qua == pred.du_doan)
            changed = True
            break
    if changed:
        db.session.commit()


@app.route("/api/leagues")
def api_leagues():
    """Trả về danh sách giải đấu để frontend hiển thị dropdown chọn giải."""
    return jsonify({
        "leagues": [
            {"key": key, "name": cfg["name"]} for key, cfg in LEAGUES.items()
        ],
        "default": DEFAULT_LEAGUE_KEY,
    })


@app.route("/api/teams")
def api_teams():
    """Trả về danh sách các đội để frontend hiển thị dropdown.
    Query param ?league=<key> chọn giải đấu, mặc định Ngoại hạng Anh."""
    league_key = _valid_league_key(request.args.get("league", DEFAULT_LEAGUE_KEY))
    try:
        _, _, teams, _ = get_model(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"teams": teams, "league": league_key})


@app.route("/api/season-table")
def api_season_table():
    """
    Bảng xếp hạng CHÍNH THỨC của mùa giải mới nhất (đúng nghĩa: thắng/hoà/
    thua/bàn thắng/bàn thua/hiệu số/điểm, chỉ tính các trận trong mùa đó) —
    khác với /api/leaderboard (tổng hợp nhiều mùa, dùng cho so sánh sức
    mạnh tổng thể của model).
    Query param ?league=<key> chọn giải đấu, mặc định Ngoại hạng Anh.
    """
    league_key = _valid_league_key(request.args.get("league", DEFAULT_LEAGUE_KEY))
    try:
        _, team_state, _, _ = get_model(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    season_table = team_state.get("season_table")
    if season_table is None:
        return jsonify({
            "error": "Model cache hiện tại chưa có dữ liệu bảng xếp hạng theo mùa "
                     "(được train bằng bản code cũ). Hãy chạy lại "
                     f"`python train_offline.py --league \"{LEAGUES[league_key]['code']}\"` "
                     "để tạo cache mới rồi dùng lại tính năng này."
        }), 409

    return jsonify({
        "league": league_key,
        "season": team_state.get("season_label"),
        "rows": season_table,
    })


@app.route("/api/leaderboard")
def api_leaderboard():
    """Bảng xếp hạng các đội của 1 giải đấu, sắp xếp theo thông số do FE chọn.
    Query params:
      ?league=<key>  giải đấu, mặc định Ngoại hạng Anh.
      ?sort=<key>    thông số sắp xếp, xem LEADERBOARD_SORT_FIELDS
                     (mặc định "diem" - điểm số)."""
    league_key = _valid_league_key(request.args.get("league", DEFAULT_LEAGUE_KEY))
    sort_by = request.args.get("sort", "diem")
    if sort_by not in LEADERBOARD_SORT_FIELDS:
        sort_by = "diem"

    try:
        _, team_state, teams, _ = get_model(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    rows = build_leaderboard(team_state, teams)
    rows = sort_leaderboard(rows, sort_by)

    return jsonify({
        "league": league_key,
        "sort": sort_by,
        "sort_options": [
            {"key": k, "label": v} for k, v in LEADERBOARD_SORT_FIELDS.items()
        ],
        "rows": rows,
    })


@app.route("/api/players")
def api_players():
    """Bảng xếp hạng rating cầu thủ của 1 giải đấu, lọc theo đội/vị trí,
    sắp xếp theo thông số do FE chọn.
    Query params:
      ?league=<key>      giải đấu, mặc định Ngoại hạng Anh.
      ?team=<tên đội>    lọc theo 1 đội (không bắt buộc).
      ?position=<FW|MF|DF|GK>  lọc theo vị trí (không bắt buộc).
      ?sort=<key>        thông số sắp xếp, xem PLAYER_SORT_FIELDS
                         (mặc định "rating")."""
    league_key = _valid_league_key(request.args.get("league", DEFAULT_LEAGUE_KEY))
    team = (request.args.get("team") or "").strip() or None
    position = (request.args.get("position") or "").strip().upper() or None
    if position not in PLAYER_POSITIONS:
        position = None
    sort_by = request.args.get("sort", "rating")
    if sort_by not in PLAYER_SORT_FIELDS:
        sort_by = "rating"

    try:
        players, meta = get_player_ratings(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    rows = build_player_leaderboard(players, team=team, position=position)
    rows = sort_player_leaderboard(rows, sort_by)

    return jsonify({
        "league": league_key,
        "season": meta.get("season_label") if meta else None,
        "sort": sort_by,
        "sort_options": [{"key": k, "label": v} for k, v in PLAYER_SORT_FIELDS.items()],
        "positions": list(PLAYER_POSITIONS),
        "rows": rows,
    })


@app.route("/api/retrain-players", methods=["POST"])
def api_retrain_players():
    """Tính lại rating cầu thủ từ đầu (tải dữ liệu mới nhất từ FBref).
    Body JSON { "league": "la-liga", "season": "2425" } không bắt buộc."""
    data = request.get_json(silent=True) or {}
    league_key = _valid_league_key(data.get("league", DEFAULT_LEAGUE_KEY))
    season = data.get("season") or _default_player_season(league_key)
    try:
        players, meta = load_or_build_player_ratings(
            league_key=league_key,
            league_code=LEAGUES[league_key]["code"],
            season=season,
            force_rebuild=True,
        )
    except Exception as e:
        logger.error("Lỗi khi retrain rating cầu thủ (%s): %s", league_key, e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    _player_state[f"{league_key}:{season}"] = {"players": players, "meta": meta}
    return jsonify({
        "message": f"Đã tính lại rating cầu thủ cho {LEAGUES[league_key]['name']} (mùa {season}).",
        "so_cau_thu": len(players),
        "season": season,
    })


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
    Body JSON mong đợi:
    { "doi_nha": "Arsenal", "doi_khach": "Chelsea", "league": "premier-league" }
    ("league" không bắt buộc, mặc định Ngoại hạng Anh, giữ tương thích code cũ)
    """
    data = request.get_json(silent=True) or {}
    doi_nha = (data.get("doi_nha") or "").strip()
    doi_khach = (data.get("doi_khach") or "").strip()
    league_key = _valid_league_key(data.get("league", DEFAULT_LEAGUE_KEY))

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
        model, team_state, _, _ = get_model(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        result = predict_match(model, team_state, doi_nha, doi_khach)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Lỗi khi dự đoán: %s", e, exc_info=True)
        return jsonify({"error": "Có lỗi xảy ra, vui lòng thử lại."}), 500

    # Cầu thủ nổi bật mỗi đội — tính năng miễn phí cho mọi người xem, không
    # chặn Premium. Nếu chưa có cache rating cầu thủ cho giải này thì bỏ
    # qua (trả None), không làm hỏng luồng dự đoán chính vốn đang chạy ổn.
    try:
        players, _ = get_player_ratings(league_key)
        result["cau_thu_noi_bat"] = {
            "nha": get_key_players(players, doi_nha, limit=3),
            "khach": get_key_players(players, doi_khach, limit=3),
        }
    except RuntimeError:
        result["cau_thu_noi_bat"] = None

    if is_logged_in:
        current_user.register_prediction()

        # Lưu lại lịch sử dự đoán để hiển thị ở trang "Lịch sử dự đoán" +
        # tính độ chính xác thật của AI theo thời gian cho từng tài khoản.
        probs_by_label = {"nha": result["thang_nha"], "hoa": result["hoa"], "khach": result["thang_khach"]}
        du_doan = max(probs_by_label, key=probs_by_label.get)
        db.session.add(Prediction(
            user_id=current_user.id,
            league=league_key,
            doi_nha=doi_nha,
            doi_khach=doi_khach,
            thang_nha=result["thang_nha"],
            hoa=result["hoa"],
            thang_khach=result["thang_khach"],
            ty_so_du_doan=(result.get("ty_so_chinh_xac") or {}).get("du_doan_nhat"),
            du_doan=du_doan,
        ))
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
    Query param ?league=<key> chọn giải đấu, mặc định Ngoại hạng Anh.
    """
    league_key = _valid_league_key(request.args.get("league", DEFAULT_LEAGUE_KEY))
    try:
        _, _, _, metrics = get_model(league_key)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(metrics)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Huấn luyện lại mô hình từ đầu (tải dữ liệu mới nhất từ FBref).
    Body JSON { "league": "la-liga" } không bắt buộc, mặc định Ngoại hạng Anh."""
    data = request.get_json(silent=True) or {}
    league_key = _valid_league_key(data.get("league", DEFAULT_LEAGUE_KEY))
    try:
        model, team_state, teams, metrics = load_or_train_model(
            force_retrain=True, league_key=league_key
        )
    except Exception as e:
        logger.error("Lỗi khi retrain (%s): %s", league_key, e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    _state[league_key] = {"model": model, "team_state": team_state, "teams": teams, "metrics": metrics}
    _backfill_predictions(league_key, team_state)
    return jsonify({
        "message": f"Đã huấn luyện lại mô hình cho {LEAGUES[league_key]['name']}.",
        "so_doi": len(teams),
        "metrics": metrics,
    })


@app.route("/api/history")
@login_required
def api_history():
    """Lịch sử dự đoán của tài khoản đang đăng nhập + độ chính xác thực tế
    của AI (chỉ tính trên các dự đoán ĐÃ xác định được kết quả thật)."""
    # Đối chiếu kết quả thật cho các dự đoán đang chờ, với TỪNG giải đấu
    # user này có dự đoán (không chỉ giải đang chọn trên FE).
    leagues_pending = {
        row.league for row in
        Prediction.query.filter_by(user_id=current_user.id, ket_qua_thuc_te=None)
        .with_entities(Prediction.league).distinct()
    }
    for league_key in leagues_pending:
        try:
            _, team_state, _, _ = get_model(league_key)
        except RuntimeError:
            continue
        _backfill_predictions(league_key, team_state)

    rows = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    decided = [r for r in rows if r.ket_qua_thuc_te is not None]
    so_dung = sum(1 for r in decided if r.dung)
    ty_le_chinh_xac = round(so_dung / len(decided) * 100, 1) if decided else None

    return jsonify({
        "rows": [r.to_dict() for r in rows],
        "thong_ke": {
            "tong_so": len(rows),
            "da_co_ket_qua": len(decided),
            "dung": so_dung,
            "sai": len(decided) - so_dung,
            "ty_le_chinh_xac": ty_le_chinh_xac,
        },
    })


@app.route("/api/challenge/current")
def api_challenge_current():
    """Danh sách các trận thử thách đang mở (chưa chốt sổ), kèm lượt đoán
    của người dùng hiện tại cho từng trận nếu đã đăng nhập và đã đoán.
    Trả về mảng (không chỉ 1 trận) để sau này hỗ trợ nhiều trận/ngày."""
    matches = (
        ChallengeMatch.query.filter_by(is_active=True, ket_qua_thuc_te=None)
        .order_by(ChallengeMatch.created_at.desc())
        .all()
    )
    my_guesses = {}
    if current_user.is_authenticated and matches:
        rows = ChallengeGuess.query.filter(
            ChallengeGuess.user_id == current_user.id,
            ChallengeGuess.match_id.in_([m.id for m in matches]),
        ).all()
        my_guesses = {r.match_id: r.du_doan for r in rows}

    out = []
    for m in matches:
        d = m.to_dict()
        d["my_guess"] = my_guesses.get(m.id)
        out.append(d)

    return jsonify({
        "matches": out,
        "logged_in": current_user.is_authenticated,
        "points": CHALLENGE_POINTS,
    })


@app.route("/api/challenge/guess", methods=["POST"])
@login_required
def api_challenge_guess():
    """Body JSON: { "match_id": 1, "du_doan": "nha" | "hoa" | "khach" }"""
    data = request.get_json(silent=True) or {}
    match_id = data.get("match_id")
    du_doan = (data.get("du_doan") or "").strip()

    if du_doan not in ("nha", "hoa", "khach"):
        return jsonify({"error": "Lượt đoán không hợp lệ."}), 400

    match = ChallengeMatch.query.get(match_id)
    if match is None or not match.is_active:
        return jsonify({"error": "Không tìm thấy trận thử thách này."}), 404
    if match.ket_qua_thuc_te is not None:
        return jsonify({"error": "Trận này đã kết thúc, không thể đoán nữa."}), 409

    existing = ChallengeGuess.query.filter_by(match_id=match.id, user_id=current_user.id).first()
    if existing is not None:
        return jsonify({"error": "Bạn đã đoán trận này rồi.", "my_guess": existing.du_doan}), 409

    db.session.add(ChallengeGuess(match_id=match.id, user_id=current_user.id, du_doan=du_doan))
    db.session.commit()

    return jsonify({
        "message": "Đã ghi nhận lượt đoán của bạn!",
        "match": match.to_dict(),
        "my_guess": du_doan,
    })


@app.route("/api/challenge/leaderboard")
def api_challenge_leaderboard():
    """Top người chơi theo tổng XP tích lũy từ thử thách dự đoán.
    Email được che bớt vì đây là danh sách công khai."""
    top = User.query.filter(User.total_xp > 0).order_by(User.total_xp.desc()).limit(20).all()
    rows = [
        {"rank": i + 1, "email": _mask_email(u.email), "total_xp": u.total_xp}
        for i, u in enumerate(top)
    ]

    my_rank = None
    if current_user.is_authenticated and current_user.total_xp > 0:
        higher = User.query.filter(User.total_xp > current_user.total_xp).count()
        my_rank = {"rank": higher + 1, "total_xp": current_user.total_xp}

    return jsonify({"rows": rows, "me": my_rank})


@app.route("/api/challenge/history")
@login_required
def api_challenge_history():
    """Các trận thử thách ĐÃ chốt sổ mà người dùng hiện tại từng đoán,
    kèm số điểm nhận được cho mỗi trận."""
    rows = (
        db.session.query(ChallengeGuess, ChallengeMatch)
        .join(ChallengeMatch, ChallengeGuess.match_id == ChallengeMatch.id)
        .filter(ChallengeGuess.user_id == current_user.id, ChallengeMatch.ket_qua_thuc_te.isnot(None))
        .order_by(ChallengeMatch.settled_at.desc())
        .limit(50)
        .all()
    )
    out = []
    for guess, match in rows:
        d = match.to_dict()
        d["my_guess"] = guess.du_doan
        d["diem"] = guess.diem
        out.append(d)
    return jsonify({"rows": out, "total_xp": current_user.total_xp})


@app.route("/health")
def health():
    """Health check endpoint, hữu ích khi deploy (Render/Railway)."""
    default_state = _state.get(DEFAULT_LEAGUE_KEY, {})
    return jsonify({
        "status": "healthy",
        "model_ready": default_state.get("model") is not None,
        "num_teams": len(default_state["teams"]) if default_state.get("teams") else 0,
        "leagues_loaded": [k for k, v in _state.items() if v.get("model") is not None],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
