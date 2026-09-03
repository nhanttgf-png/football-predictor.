"""
migrate_add_total_xp.py
------------------------
Chạy 1 LẦN DUY NHẤT sau khi pull code có tính năng "Thử thách dự đoán" về,
NẾU bạn đã có sẵn file app.db từ trước (đã có user đăng ký).

db.create_all() của SQLAlchemy chỉ tạo bảng MỚI, không tự thêm cột mới vào
bảng users đã tồn tại -> gây lỗi "no such column: users.total_xp".
Script này thêm cột đó bằng ALTER TABLE, an toàn để chạy nhiều lần (tự bỏ
qua nếu cột đã tồn tại rồi).

Cách chạy (đứng tại thư mục project, chỗ có file app.db):
    python migrate_add_total_xp.py
Nếu app.db của bạn ở đường dẫn khác, sửa DB_PATH bên dưới.
"""

import sqlite3
import os

DB_PATH = "app.db"

if not os.path.exists(DB_PATH):
    raise SystemExit(f'Không tìm thấy "{DB_PATH}" ở thư mục hiện tại. '
                      f'Hãy cd vào đúng thư mục project rồi chạy lại.')

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("PRAGMA table_info(users)")
existing_cols = {row[1] for row in cur.fetchall()}

if "total_xp" in existing_cols:
    print('Cột "total_xp" đã tồn tại trong bảng users -- không cần làm gì thêm.')
else:
    cur.execute("ALTER TABLE users ADD COLUMN total_xp INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    print('Đã thêm cột "total_xp" (mặc định 0) vào bảng users thành công!')

conn.close()
