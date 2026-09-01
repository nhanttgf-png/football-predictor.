"""
train_offline.py
-----------------
CHẠY FILE NÀY Ở MÁY LOCAL CỦA BẠN (không chạy trên Render).

soccerdata cần một trình duyệt Chrome thật để lấy dữ liệu từ FBref, nhưng máy
chủ deploy miễn phí (Render, Railway...) không có sẵn Chrome. Vì vậy ta tải
dữ liệu và huấn luyện model ngay tại máy mình (nơi có Chrome), lưu kết quả
vào model_cache.pkl, rồi đẩy file này lên GitHub luôn.

Khi đó, trên server, app.py sẽ chỉ đọc model_cache.pkl có sẵn, KHÔNG cần
soccerdata/Chrome hoạt động trên đó nữa.

Cách dùng:
    python train_offline.py

Xong thì:
    git add model_cache.pkl
    git commit -m "Cap nhat model cache"
    git push
"""

import pickle

from model import CACHE_PATH, train_model

print("Đang tải dữ liệu từ FBref và huấn luyện mô hình (cần Chrome trên máy này)...")
result = train_model()

with open(CACHE_PATH, "wb") as f:
    pickle.dump(result, f)

_, _, _, teams = result
print(f"Xong! Đã lưu model_cache.pkl với dữ liệu của {len(teams)} đội.")
print("Giờ bạn có thể: git add model_cache.pkl, commit, rồi push lên GitHub.")