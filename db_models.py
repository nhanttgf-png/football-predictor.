"""
db_models.py
------------
Model dữ liệu cho tài khoản người dùng (User), lưu vào SQLite qua
SQLAlchemy. Trạng thái Premium + số lượt dự đoán miễn phí đã dùng nằm
NGAY TRONG DB — thay cho bộ nhớ tạm (in-memory dict) ở bản trước — nên:
  - Sống sót qua việc restart server (khác với dict trong RAM).
  - Đúng khi chạy nhiều worker (gunicorn -w > 1) hoặc nhiều instance, vì
    mọi worker/instance đọc chung một chỗ (1 file SQLite, hoặc đổi sang
    Postgres khi deploy thật bằng cách đổi SQLALCHEMY_DATABASE_URI).

Không có tên "models.py" để tránh nhầm với model.py (logic AI/ML) đã có
sẵn trong project.
"""

from datetime import date, datetime

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ---- Trạng thái Premium (được payments.py cập nhật sau khi Stripe
    # xác nhận thanh toán qua webhook — KHÔNG bao giờ set trực tiếp từ FE) ----
    is_premium = db.Column(db.Boolean, default=False, nullable=False)
    premium_since = db.Column(db.DateTime, nullable=True)
    stripe_customer_id = db.Column(db.String(255), nullable=True, index=True)
    stripe_subscription_id = db.Column(db.String(255), nullable=True)

    # ---- Giới hạn lượt dùng miễn phí / ngày ----
    free_predictions_used = db.Column(db.Integer, default=0, nullable=False)
    usage_date = db.Column(db.String(10), default=lambda: date.today().isoformat())

    # ---- Thử thách dự đoán (Prediction Challenge): tổng điểm XP cộng dồn
    # từ các lượt đoán kết quả (thắng/hòa/thua) đúng ở các trận challenge,
    # xem ChallengeMatch/ChallengeGuess bên dưới. ----
    total_xp = db.Column(db.Integer, default=0, nullable=False)

    # -------- Mật khẩu --------
    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    # -------- Giới hạn lượt dùng --------
    def _reset_if_new_day(self) -> None:
        today = date.today().isoformat()
        if self.usage_date != today:
            self.usage_date = today
            self.free_predictions_used = 0

    def can_predict(self, free_limit: int) -> bool:
        self._reset_if_new_day()
        return self.is_premium or self.free_predictions_used < free_limit

    def register_prediction(self) -> None:
        self._reset_if_new_day()
        self.free_predictions_used += 1

    def usage_dict(self, free_limit: int) -> dict:
        self._reset_if_new_day()
        remaining = None if self.is_premium else max(0, free_limit - self.free_predictions_used)
        return {
            "limit": free_limit,
            "used": self.free_predictions_used,
            "remaining": remaining,
            "premium": self.is_premium,
        }


class Prediction(db.Model):
    """Lịch sử dự đoán của 1 tài khoản — dùng cho tính năng "Prediction
    History": lưu lại dự đoán AI, và sau này đối chiếu với kết quả thật
    (nếu trận đấu đã diễn ra và có trong dữ liệu huấn luyện mới) để tính
    độ chính xác thực tế của AI cho người dùng xem.

    `ket_qua_thuc_te` để trống (None) cho tới khi trận đấu đó xuất hiện
    trong dữ liệu lịch sử (xem app._backfill_predictions) — nghĩa là
    "chưa có kết quả / trận chưa đá xong theo dữ liệu hiện có".
    """
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    league = db.Column(db.String(50), nullable=False)
    doi_nha = db.Column(db.String(120), nullable=False)
    doi_khach = db.Column(db.String(120), nullable=False)

    thang_nha = db.Column(db.Float, nullable=False)
    hoa = db.Column(db.Float, nullable=False)
    thang_khach = db.Column(db.Float, nullable=False)
    ty_so_du_doan = db.Column(db.String(10), nullable=True)

    # "nha" | "hoa" | "khach" -- kết quả AI cho là khả dĩ nhất lúc dự đoán
    du_doan = db.Column(db.String(10), nullable=False)
    # "nha" | "hoa" | "khach" | None (chưa xác định được kết quả thật)
    ket_qua_thuc_te = db.Column(db.String(10), nullable=True)
    dung = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "league": self.league,
            "doi_nha": self.doi_nha,
            "doi_khach": self.doi_khach,
            "thang_nha": self.thang_nha,
            "hoa": self.hoa,
            "thang_khach": self.thang_khach,
            "ty_so_du_doan": self.ty_so_du_doan,
            "du_doan": self.du_doan,
            "ket_qua_thuc_te": self.ket_qua_thuc_te,
            "dung": self.dung,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==================== THỬ THÁCH DỰ ĐOÁN (Prediction Challenge) ====================
#
# Khác với `Prediction` ở trên (log MỌI lượt tra cứu tỷ lệ, dùng cho lịch
# sử cá nhân + đo độ chính xác AI), 2 bảng dưới đây phục vụ tính năng
# "chơi" riêng: admin chọn ra 1 (hoặc vài) trận "thử thách", người dùng
# đoán TRƯỚC kết quả (thắng/hòa/thua) — không xem trước dự đoán AI có sẵn
# nào khác — rồi khi trận đấu kết thúc ngoài đời, admin nhập kết quả thật
# vào để "chốt sổ" (settle), hệ thống tự cộng XP cho ai đoán đúng.
#
# Vì app này KHÔNG có nguồn lịch thi đấu/kết quả trực tiếp (dữ liệu chỉ
# refresh khi chạy train_offline.py thủ công), việc "trận đấu kết thúc"
# ở đây được XÁC NHẬN THỦ CÔNG bởi admin qua /api/admin/challenge/settle,
# thay vì tự động ngay khi trọng tài thổi còi.

class ChallengeMatch(db.Model):
    __tablename__ = "challenge_matches"

    id = db.Column(db.Integer, primary_key=True)
    league = db.Column(db.String(50), nullable=False)
    doi_nha = db.Column(db.String(120), nullable=False)
    doi_khach = db.Column(db.String(120), nullable=False)

    # Dự đoán của AI tại thời điểm tạo trận thử thách (để so sánh với lượt
    # đoán của người dùng) -- không đổi kể cả khi model được retrain sau đó.
    ai_thang_nha = db.Column(db.Float, nullable=True)
    ai_hoa = db.Column(db.Float, nullable=True)
    ai_thang_khach = db.Column(db.Float, nullable=True)
    ai_du_doan = db.Column(db.String(10), nullable=True)  # "nha" | "hoa" | "khach"

    # "nha" | "hoa" | "khach" | None (chưa có kết quả thật / chưa chốt sổ)
    ket_qua_thuc_te = db.Column(db.String(10), nullable=True)
    ty_so_thuc_te = db.Column(db.String(10), nullable=True)  # vd "2-1", chỉ để hiển thị

    is_active = db.Column(db.Boolean, default=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    settled_at = db.Column(db.DateTime, nullable=True)

    guesses = db.relationship("ChallengeGuess", backref="match", lazy="dynamic",
                               cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "league": self.league,
            "doi_nha": self.doi_nha,
            "doi_khach": self.doi_khach,
            "ai_thang_nha": self.ai_thang_nha,
            "ai_hoa": self.ai_hoa,
            "ai_thang_khach": self.ai_thang_khach,
            "ai_du_doan": self.ai_du_doan,
            "ket_qua_thuc_te": self.ket_qua_thuc_te,
            "ty_so_thuc_te": self.ty_so_thuc_te,
            "is_active": self.is_active,
            "so_luot_doan": self.guesses.count(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChallengeGuess(db.Model):
    """1 lượt đoán kết quả (thắng/hòa/thua) của 1 tài khoản cho 1 trận
    thử thách. Mỗi tài khoản chỉ được đoán 1 lần / trận (unique constraint)."""
    __tablename__ = "challenge_guesses"
    __table_args__ = (
        db.UniqueConstraint("match_id", "user_id", name="uq_challenge_guess_user_match"),
    )

    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey("challenge_matches.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # "nha" | "hoa" | "khach" -- lượt đoán của người dùng
    du_doan = db.Column(db.String(10), nullable=False)
    # None cho tới khi trận được chốt sổ; sau đó là số XP nhận được (0 nếu đoán sai)
    diem = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "match_id": self.match_id,
            "du_doan": self.du_doan,
            "diem": self.diem,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChallengeSeason(db.Model):
    """1 dòng DUY NHẤT (id=1) lưu thông tin "mùa" hiện tại của bảng xếp
    hạng XP -- không phải mùa giải bóng đá, mà là 1 chu kỳ tính điểm XP:
    admin có thể "Reset" (đưa XP mọi người về 0, bắt đầu mùa mới) hoặc
    "Gia hạn" (chỉ đổi ngày kết thúc, giữ nguyên XP đang có).
    ends_at = None nghĩa là không giới hạn thời gian (chạy tới khi admin
    tự tay reset)."""
    __tablename__ = "challenge_season"

    id = db.Column(db.Integer, primary_key=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ends_at = db.Column(db.DateTime, nullable=True)

    @classmethod
    def current(cls):
        """Lấy dòng duy nhất (id=1), tự tạo nếu chưa có (mùa đầu tiên,
        không giới hạn thời gian)."""
        row = cls.query.get(1)
        if row is None:
            row = cls(id=1)
            db.session.add(row)
            db.session.commit()
        return row

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
        }


class PenaltyGameScore(db.Model):
    """Một ván Penalty Shootout đã gửi điểm từ client.
    XP của game được cộng vào cùng total_xp với Prediction Challenge.
    """
    __tablename__ = "penalty_game_scores"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    goals = db.Column(db.Integer, nullable=False)
    xp = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship("User", backref=db.backref("penalty_scores", lazy="dynamic"))
