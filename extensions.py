"""
extensions.py
-------------
Khởi tạo các extension dùng chung (SQLAlchemy, Flask-Login) ở MỘT chỗ duy
nhất. Nếu khởi tạo trực tiếp trong app.py rồi import ngược lại từ
db_models.py / auth.py / payments.py sẽ dễ dính lỗi import vòng (circular
import) vì app.py cũng import các file đó. Tách ra file riêng, không import
gì khác, giải quyết gọn vấn đề này.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
