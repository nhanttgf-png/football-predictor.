# Dự đoán tỷ lệ bóng đá

Web app dự đoán tỷ lệ thắng/hòa/thua cho các trận đấu Premier League, dựa trên
mô hình Random Forest học từ dữ liệu FBref (mùa 2023/24).

## Cấu trúc project

```
football-predictor/
├── app.py                    # Flask backend + API
├── model.py                  # Logic AI: tải dữ liệu, huấn luyện, dự đoán kết quả trận đấu
├── player_ratings.py         # Tính rating cầu thủ (0-10) từ số liệu FBref
├── train_offline.py          # Script train model dự đoán trận đấu (chạy local, có Chrome)
├── train_players_offline.py  # Script tính rating cầu thủ (chạy local, có Chrome)
├── debug_team_stats.py       # Soi số liệu Elo/phong độ 1 đội từ cache
├── debug_player_stats.py     # Soi rating cầu thủ + chỉ số nhận diện được từ cache
├── requirements.txt          # Thư viện Python cần cài
├── templates/
│   └── index.html            # Trang web chính
├── static/
│   ├── style.css             # Giao diện
│   └── script.js             # Gọi API, cập nhật giao diện
├── model_cache*.pkl          # (tự sinh ra) cache model dự đoán, 1 file/giải đấu
└── player_ratings_*.pkl      # (tự sinh ra) cache rating cầu thủ, 1 file/giải đấu
```

## Chạy thử ở máy local

1. Tạo môi trường ảo (khuyên dùng, để không đụng vào Python hệ thống):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```

3. Chạy server:
   ```bash
   python app.py
   ```

4. Mở trình duyệt: http://127.0.0.1:5000

> Lần chạy đầu tiên sẽ hơi lâu vì phải tải dữ liệu từ FBref về và huấn luyện
> mô hình. Lần sau sẽ nhanh hơn nhờ file `model_cache.pkl`. Muốn huấn luyện
> lại dữ liệu mới nhất thì gọi `POST /api/retrain` hoặc xoá file cache đó.

## API có sẵn

| Method | Endpoint               | Mô tả                                    |
|--------|------------------------|-------------------------------------------|
| GET    | `/api/teams`           | Trả về danh sách các đội                  |
| POST   | `/api/predict`         | Body: `{"doi_nha": "...", "doi_khach": "..."}` — kết quả có kèm `cau_thu_noi_bat` (3 cầu thủ rating cao nhất mỗi đội, nếu đã có cache rating) |
| POST   | `/api/retrain`         | Huấn luyện lại mô hình dự đoán từ đầu     |
| GET    | `/api/players`         | Bảng xếp hạng rating cầu thủ. Query: `?league=&team=&position=FW\|MF\|DF\|GK&sort=` |
| POST   | `/api/retrain-players` | Tính lại rating cầu thủ từ đầu            |

## Tính năng rating cầu thủ

Rating (thang 0-10) được tính từ số liệu MÙA GIẢI trên FBref (bàn, kiến
tạo, xG, chuyền bóng, phòng ngự, cản phá thủ môn...), xếp hạng phần trăm
so với các cầu thủ khác cùng giải rồi lấy trung bình có trọng số THEO VỊ
TRÍ (tiền đạo nặng tấn công, hậu vệ nặng phòng ngự, thủ môn dùng bộ chỉ
số riêng). Cầu thủ đá dưới ~5 trận (450 phút) không được xếp rating vì
mẫu quá nhỏ dễ gây nhiễu. Chi tiết công thức + cách tuỳ chỉnh nằm trong
`player_ratings.py`.

Cũng như model dự đoán trận đấu, **FBref cần trình duyệt (Chrome) để cào
dữ liệu nên Render KHÔNG tự tính được** — phải chạy ở máy local:

```bash
python train_players_offline.py                       # Ngoại hạng Anh, mùa gần nhất đã đá đủ
python train_players_offline.py --all                  # tất cả giải đang hỗ trợ
python train_players_offline.py --league "ESP-La Liga" --season 2425
```

Sau đó commit + push các file `player_ratings_<giải>.pkl` mới lên GitHub.
Dùng `python debug_player_stats.py --league la-liga --top 20` để soi kết
quả và xem chỉ số nào chưa nhận diện được cột dữ liệu (nếu soccerdata đổi
tên cột, chỉnh lại từ khoá tìm kiếm trong `METRICS` ở `player_ratings.py`).

## Làm việc nhóm trên GitHub (2-3 người)

1. Một bạn tạo repo trên GitHub, push code này lên nhánh `main`.
2. Hai bạn còn lại `git clone` repo về máy.
3. Mỗi khi làm tính năng mới, tạo nhánh riêng thay vì code thẳng trên `main`:
   ```bash
   git checkout -b ten-nhanh-cua-ban
   ```
4. Commit và push nhánh đó lên GitHub:
   ```bash
   git add .
   git commit -m "Mô tả ngắn gọn thay đổi"
   git push origin ten-nhanh-cua-ban
   ```
5. Vào GitHub tạo **Pull Request** để bạn còn lại review trước khi merge vào `main`.
6. Tránh 2 người cùng sửa 1 file cùng lúc trên `main` để đỡ bị conflict.

Gợi ý chia việc:
- 1 người phụ trách `model.py` (cải thiện mô hình, thêm feature).
- 1 người phụ trách `app.py` (API, xử lý lỗi, thêm giải đấu khác).
- 1 người phụ trách `templates/` + `static/` (giao diện, trải nghiệm người dùng).

## Ý tưởng phát triển thêm

- Thêm nhiều giải đấu / mùa giải khác để chọn.
- Thêm biểu đồ lịch sử đối đầu (head-to-head) giữa 2 đội.
- Deploy lên Render, Railway hoặc PythonAnywhere để bạn bè truy cập được từ xa.
- Thêm test cho `model.py` bằng `pytest`.
