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
