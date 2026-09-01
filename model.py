"""
model.py
--------
Chứa toàn bộ logic AI: tải dữ liệu, huấn luyện mô hình, và dự đoán.
Đây là phần "bộ não" được tách ra từ file gốc "tỉ lệ bóng đá.py" của bạn,
để app.py (Flask) có thể gọi lại nhiều lần mà không cần copy code lung tung.
"""

import os
import pickle
import numpy as np
import pandas as pd
import soccerdata as sd
from sklearn.ensemble import RandomForestClassifier

# Nơi lưu cache mô hình đã huấn luyện, để lần chạy sau không cần tải lại dữ liệu
# (tải dữ liệu từ FBref khá chậm, nên cache lại cho đỡ chờ mỗi lần restart server)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "model_cache.pkl")

LEAGUE = "ENG-Premier League"
SEASON = "2324"


def _get_result(row):
    """2 = đội nhà thắng, 1 = hòa, 0 = đội khách thắng"""
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] == row["away_score"]:
        return 1
    return 0


def train_model(league: str = LEAGUE, season: str = SEASON):
    """
    Tải dữ liệu lịch sử từ FBref và huấn luyện lại mô hình từ đầu.
    Trả về (model, home_stats, away_stats, danh_sach_doi).
    """
    fbref = sd.FBref(leagues=league, seasons=season)
    games = fbref.read_schedule()

    completed_games = games[games["game_id"].notnull()].copy()
    completed_games[["home_score", "away_score"]] = (
        completed_games["score"]
        .str.split("–|-", regex=True, expand=True)
        .astype(float)
    )

    completed_games["target"] = completed_games.apply(_get_result, axis=1)

    home_stats = completed_games.groupby("home_team")["home_score"].mean()
    away_stats = completed_games.groupby("away_team")["away_score"].mean()

    completed_games["home_form"] = completed_games["home_team"].map(home_stats)
    completed_games["away_form"] = completed_games["away_team"].map(away_stats)

    X = completed_games[["home_form", "away_form"]]
    y = completed_games["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    teams = sorted(set(home_stats.index) | set(away_stats.index))

    return model, home_stats, away_stats, teams


def load_or_train_model(force_retrain: bool = False):
    """
    Nếu đã có cache thì load lên cho nhanh, không thì huấn luyện mới rồi lưu cache.
    """
    if not force_retrain and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    result = train_model()

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)

    return result


def predict_match(model, home_stats, away_stats, doi_nha: str, doi_khach: str):
    """
    Dự đoán tỉ lệ thắng/hòa/thua cho 1 trận đấu.
    Trả về dict kết quả, hoặc raise ValueError nếu tên đội không hợp lệ.
    """
    if doi_nha not in home_stats.index:
        raise ValueError(f'Không tìm thấy đội nhà "{doi_nha}" trong dữ liệu.')
    if doi_khach not in away_stats.index:
        raise ValueError(f'Không tìm thấy đội khách "{doi_khach}" trong dữ liệu.')

    phong_do_home = home_stats[doi_nha]
    phong_do_away = away_stats[doi_khach]

    match_data = np.array([[phong_do_home, phong_do_away]])
    probs = model.predict_proba(match_data)[0]

    return {
        "doi_nha": doi_nha,
        "doi_khach": doi_khach,
        "thang_nha": round(float(probs[2]) * 100, 1),
        "hoa": round(float(probs[1]) * 100, 1),
        "thang_khach": round(float(probs[0]) * 100, 1),
    }
