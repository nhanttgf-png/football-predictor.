"""
model.py
--------
Chứa toàn bộ logic AI: tải dữ liệu, huấn luyện mô hình, và dự đoán.

Phiên bản hoàn chỉnh với nhiều cải tiến:
1. Không data leakage - chỉ dùng dữ liệu quá khứ
2. Nhiều feature engineering hơn (xG, form, H2H, rest days...)
3. Ensemble nhiều model (RandomForest, XGBoost, LightGBM, MLP)
4. Hyperparameter tuning với TimeSeriesSplit
5. Feature selection tự động
6. Calibration cho xác suất chính xác
7. Cache thông minh với timestamp
"""

import os
import pickle
import time
import math
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import soccerdata as sd
from sklearn.ensemble import (
    RandomForestClassifier, 
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Optional imports - sẽ không lỗi nếu chưa cài
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost không được cài đặt, sẽ bỏ qua model này")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LightGBM không được cài đặt, sẽ bỏ qua model này")

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==================== CẤU HÌNH ====================

CACHE_DIR = os.path.dirname(__file__)
CACHE_MAX_AGE_SECONDS = 3 * 24 * 60 * 60  # 3 ngày

# Danh sách mùa giải (dùng chung cho tất cả các giải đấu bên dưới)
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]

# Giữ lại LEAGUE/CACHE_PATH để tương thích code cũ (train_offline.py bản cũ,
# debug_team_stats.py...) — mặc định trỏ về Ngoại hạng Anh.
LEAGUE = "ENG-Premier League"
CACHE_PATH = os.path.join(CACHE_DIR, "model_cache.pkl")

# ==================== CÁC GIẢI ĐẤU ĐƯỢC HỖ TRỢ ====================
# key: dùng trong URL/API (?league=la-liga) và trong tên file cache.
# code: mã giải đấu mà soccerdata/FBref dùng để tải dữ liệu.
# name: tên hiển thị trên web.
LEAGUES = {
    "premier-league": {"name": "Ngoại hạng Anh", "code": "ENG-Premier League", "seasons": SEASONS},
    "la-liga":        {"name": "La Liga",         "code": "ESP-La Liga",       "seasons": SEASONS},
    "serie-a":        {"name": "Serie A",         "code": "ITA-Serie A",       "seasons": SEASONS},
    "bundesliga":     {"name": "Bundesliga",       "code": "GER-Bundesliga",   "seasons": SEASONS},
    "ligue-1":        {"name": "Ligue 1",          "code": "FRA-Ligue 1",      "seasons": SEASONS},
}
DEFAULT_LEAGUE_KEY = "premier-league"


def _cache_path(league_key: str) -> str:
    """Mỗi giải đấu có 1 file cache riêng (model_cache_<league>.pkl), tránh
    train xong giải này lại đè mất cache của giải khác."""
    if league_key == DEFAULT_LEAGUE_KEY:
        # Giữ đúng tên file cũ cho giải mặc định, để không phải train lại
        # từ đầu Premier League khi nâng cấp lên bản nhiều giải đấu này.
        return CACHE_PATH
    return os.path.join(CACHE_DIR, f"model_cache_{league_key}.pkl")

# Elo parameters
ELO_K = 30  # Tăng K để cập nhật nhanh hơn
ELO_HOME_ADVANTAGE = 65  # Lợi thế sân nhà
ELO_START = 1500

# Form parameters
FORM_WINDOW = 5  # Số trận gần nhất cho form
H2H_WINDOW = 5   # Số trận đối đầu gần nhất

# Feature columns
FEATURE_COLS = [
    # Elo và rating
    "elo_diff",
    "elo_home",
    "elo_away",
    
    # Form tấn công và phòng ngự
    "home_form",
    "away_form",
    "home_conceded",
    "away_conceded",
    "goal_diff_form",
    
    # Form riêng sân nhà/sân khách
    "home_form_home_games",
    "away_form_away_games",
    
    # Head-to-head
    "h2h_diff",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    
    # Số ngày nghỉ
    "rest_days_diff",
    "home_rest_days",
    "away_rest_days",
    
    # Phong độ gần đây (win rate)
    "home_win_rate",
    "away_win_rate",
    "home_unbeaten_streak",
    "away_unbeaten_streak",
    
    # Sức mạnh tương đối
    "home_points_per_game",
    "away_points_per_game",
    "points_diff",
    
    # Thời điểm mùa giải
    "matchday",
    "is_start_season",
    "is_end_season",
    
    # Tỷ lệ ghi bàn
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
]

# ==================== HELPER FUNCTIONS ====================

def _get_result(row):
    """2 = đội nhà thắng, 1 = hòa, 0 = đội khách thắng"""
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] == row["away_score"]:
        return 1
    return 0


def _get_points(result: int) -> Tuple[float, float]:
    """Trả về (điểm đội nhà, điểm đội khách) dựa trên kết quả"""
    if result == 2:  # Nhà thắng
        return 3.0, 0.0
    elif result == 1:  # Hòa
        return 1.0, 1.0
    return 0.0, 3.0  # Khách thắng


def _load_raw_games(league: str, seasons: List[str]) -> pd.DataFrame:
    """
    Tải dữ liệu từ FBref với xử lý lỗi tốt hơn
    """
    logger.info(f"Đang tải dữ liệu từ FBref cho {league}, mùa {seasons}...")
    
    try:
        fbref = sd.FBref(leagues=league, seasons=seasons)
        games = fbref.read_schedule().reset_index()
        
        # Filter chỉ lấy các trận đã đá
        completed = games[games["game_id"].notnull()].copy()
        
        if len(completed) == 0:
            logger.warning("Không tìm thấy trận đấu nào!")
            return pd.DataFrame()
        
        # Parse scores
        try:
            scores = completed["score"].str.split("–|-", regex=True, expand=True)
            completed["home_score"] = scores[0].astype(float)
            completed["away_score"] = scores[1].astype(float)
        except Exception as e:
            logger.error(f"Lỗi parse scores: {e}")
            # Thử cách khác
            completed[["home_score", "away_score"]] = (
                completed["score"].str.extract(r"(\d+)[–-](\d+)").astype(float)
            )
        
        # Drop các trận có score NaN
        completed = completed.dropna(subset=["home_score", "away_score"])
        
        # Tính target
        completed["target"] = completed.apply(_get_result, axis=1)
        
        # Chuyển date thành datetime
        completed["date"] = pd.to_datetime(completed["date"])
        
        # Sắp xếp theo thời gian
        completed = completed.sort_values("date").reset_index(drop=True)
        
        logger.info(f"Tải xong {len(completed)} trận đã đá")
        return completed
        
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        raise

def _detect_season_col(games: pd.DataFrame) -> Optional[str]:
    """Tìm tên cột lưu mùa giải trong DataFrame trận đấu (khác nhau tuỳ
    phiên bản soccerdata: "season", "Season" hoặc "year")."""
    for candidate in ("season", "Season", "year"):
        if candidate in games.columns:
            return candidate
    return None


def _latest_season_slice(
    games: pd.DataFrame, seasons: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame]]:
    """
    Trả về (season_col, latest_season, df chỉ chứa các trận của mùa mới nhất).
    Nếu không xác định được cột/giá trị mùa giải hợp lệ, trả về
    (season_col, None, None) để nơi gọi tự fallback.
    """
    if not seasons:
        return None, None, None

    season_col = _detect_season_col(games)
    if season_col is None:
        return None, None, None

    latest_season = seasons[-1]
    season_df = games[games[season_col].astype(str) == str(latest_season)]
    if len(season_df) == 0:
        return season_col, None, None

    return season_col, latest_season, season_df


def _get_current_season_teams(games: pd.DataFrame, seasons: List[str]) -> List[str]:
    """
    Trả về danh sách các đội đang thi đấu ở mùa giải MỚI NHẤT (phần tử cuối
    của SEASONS), để dropdown chọn đội trên web không hiện các đội đã xuống
    hạng từ những mùa cũ (Watford, Stoke City, Cardiff City...).

    Nếu vì lý do gì không xác định được cột mùa giải, sẽ fallback về TOÀN BỘ
    đội từ trước tới nay (an toàn, chỉ là dropdown dài hơn, không lỗi).
    """
    season_col, latest_season, season_df = _latest_season_slice(games, seasons)

    if season_col is None:
        logger.warning(
            "Không tìm thấy cột mùa giải trong dữ liệu (các cột có sẵn: %s). "
            "Tạm dùng toàn bộ đội từ trước tới nay cho dropdown.",
            list(games.columns),
        )
        return sorted(set(games["home_team"]) | set(games["away_team"]))

    if season_df is None:
        logger.warning(
            "Không tìm thấy trận nào của mùa mới nhất (%s). "
            "Tạm dùng toàn bộ đội từ trước tới nay cho dropdown.",
            seasons[-1],
        )
        return sorted(set(games["home_team"]) | set(games["away_team"]))

    logger.info(
        "Mùa hiện tại (%s) có %d đội: %s",
        latest_season, len(set(season_df["home_team"]) | set(season_df["away_team"])),
        sorted(set(season_df["home_team"]) | set(season_df["away_team"])),
    )
    return sorted(set(season_df["home_team"]) | set(season_df["away_team"]))

def _build_season_standings(season_games: pd.DataFrame) -> List[Dict]:
    """
    Tính bảng xếp hạng CHÍNH THỨC (đúng nghĩa) của 1 mùa giải cụ thể, chỉ
    từ các trận của MÙA ĐÓ — khác với build_leaderboard() (dùng dữ liệu
    tích luỹ nhiều mùa cho mục đích feature engineering của model).

    season_games cần có các cột: home_team, away_team, home_score,
    away_score, target (2=nhà thắng, 1=hoà, 0=khách thắng).

    Xếp hạng theo chuẩn bóng đá: Điểm giảm dần -> Hiệu số giảm dần ->
    Bàn thắng giảm dần.
    """
    stats: Dict[str, Dict[str, float]] = {}

    def _row(team: str) -> Dict[str, float]:
        return stats.setdefault(team, {
            "so_tran": 0, "thang": 0, "hoa": 0, "thua": 0,
            "bt": 0, "bb": 0, "diem": 0,
        })

    for _, g in season_games.iterrows():
        home, away = g["home_team"], g["away_team"]
        hs, as_ = g["home_score"], g["away_score"]
        h, a = _row(home), _row(away)

        h["so_tran"] += 1
        a["so_tran"] += 1
        h["bt"] += hs
        h["bb"] += as_
        a["bt"] += as_
        a["bb"] += hs

        if g["target"] == 2:  # nhà thắng
            h["thang"] += 1
            h["diem"] += 3
            a["thua"] += 1
        elif g["target"] == 0:  # khách thắng
            a["thang"] += 1
            a["diem"] += 3
            h["thua"] += 1
        else:  # hoà
            h["hoa"] += 1
            a["hoa"] += 1
            h["diem"] += 1
            a["diem"] += 1

    rows = []
    for team, s in stats.items():
        rows.append({
            "doi": team,
            "so_tran": int(s["so_tran"]),
            "thang": int(s["thang"]),
            "hoa": int(s["hoa"]),
            "thua": int(s["thua"]),
            "bt": int(s["bt"]),
            "bb": int(s["bb"]),
            "hs": int(s["bt"] - s["bb"]),
            "diem": int(s["diem"]),
        })

    rows.sort(key=lambda r: (r["diem"], r["hs"], r["bt"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["hang"] = i

    return rows


def _safe_mean(arr: List[float], default: float = 0.0) -> float:
    """Tính mean an toàn, trả về default nếu list rỗng"""
    return float(np.mean(arr)) if arr else default


# ==================== FEATURE ENGINEERING ====================

def _build_features(games: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Xây dựng features với đầy đủ thông tin, không data leakage.
    
    Returns:
        df: DataFrame với features
        team_state: Dict chứa state cuối cùng của các đội
    """
    logger.info("Đang xây dựng features...")
    
    # State dictionaries
    elo = {}
    home_goals_for_hist = {}
    away_goals_for_hist = {}
    home_goals_against_hist = {}
    away_goals_against_hist = {}
    
    # Form riêng cho sân nhà/sân khách
    home_goals_home_games = {}
    away_goals_away_games = {}
    
    # Lịch sử đối đầu
    h2h_hist = {}
    
    # Ngày đá gần nhất
    last_played = {}
    
    # Thống kê tích lũy
    home_points = {}
    away_points = {}
    home_games_played = {}
    away_games_played = {}
    
    # Chuỗi bất bại
    home_unbeaten = {}
    away_unbeaten = {}
    
    # Tổng số bàn thắng/bại
    home_total_scored = {}
    away_total_scored = {}
    home_total_conceded = {}
    away_total_conceded = {}

    # Lịch sử trận gần đây theo từng đội (bất kể sân nhà/khách) — dùng để vẽ
    # "biểu đồ phong độ" (chuỗi W/D/L) trên FE. Chỉ giữ lại vài trận gần
    # nhất mỗi đội khi lưu vào team_state cuối cùng (xem bên dưới).
    recent_matches = {}

    rows = []
    
    for idx, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]
        match_date = g["date"]
        
        # ============ LẤY FEATURES TỪ QUÁ KHỨ ============
        
        # Elo
        home_elo = elo.get(home, ELO_START)
        away_elo = elo.get(away, ELO_START)
        
        # Form chung (5 trận gần nhất)
        home_form = _safe_mean(home_goals_for_hist.get(home, [])[-FORM_WINDOW:], 1.3)
        away_form = _safe_mean(away_goals_for_hist.get(away, [])[-FORM_WINDOW:], 1.1)
        home_conceded = _safe_mean(home_goals_against_hist.get(home, [])[-FORM_WINDOW:], 1.1)
        away_conceded = _safe_mean(away_goals_against_hist.get(away, [])[-FORM_WINDOW:], 1.3)
        
        # Form riêng sân nhà/sân khách
        home_form_home = _safe_mean(home_goals_home_games.get(home, [])[-FORM_WINDOW:], 1.5)
        away_form_away = _safe_mean(away_goals_away_games.get(away, [])[-FORM_WINDOW:], 1.0)
        
        # Head-to-head
        pair_key = frozenset((home, away))
        past_meetings = h2h_hist.get(pair_key, [])[-H2H_WINDOW:]
        
        if past_meetings:
            oriented_diffs = [
                diff if past_home == home else -diff
                for past_home, diff, _ in past_meetings
            ]
            h2h_diff = float(np.mean(oriented_diffs))
            
            h2h_home_wins = sum(1 for _, diff, _ in past_meetings if diff > 0)
            h2h_away_wins = sum(1 for _, diff, _ in past_meetings if diff < 0)
            h2h_draws = sum(1 for _, diff, _ in past_meetings if diff == 0)
        else:
            h2h_diff = 0.0
            h2h_home_wins = h2h_away_wins = h2h_draws = 0
        
        # Số ngày nghỉ
        home_rest = (match_date - last_played[home]).days if home in last_played else 7
        away_rest = (match_date - last_played[away]).days if away in last_played else 7
        rest_days_diff = float(np.clip(home_rest - away_rest, -14, 14))
        
        # Win rate và points
        home_gp = home_games_played.get(home, 0)
        away_gp = away_games_played.get(away, 0)
        
        home_win_rate = home_points.get(home, 0) / (home_gp * 3) if home_gp > 0 else 0.35
        away_win_rate = away_points.get(away, 0) / (away_gp * 3) if away_gp > 0 else 0.30
        
        home_ppg = home_points.get(home, 0) / home_gp if home_gp > 0 else 1.3
        away_ppg = away_points.get(away, 0) / away_gp if away_gp > 0 else 1.1
        
        # Chuỗi bất bại
        home_streak = home_unbeaten.get(home, 0)
        away_streak = away_unbeaten.get(away, 0)
        
        # Tỷ lệ ghi bàn trung bình
        home_scored_avg = home_total_scored.get(home, 0) / home_gp if home_gp > 0 else 1.3
        away_scored_avg = away_total_scored.get(away, 0) / away_gp if away_gp > 0 else 1.1
        home_conceded_avg = home_total_conceded.get(home, 0) / home_gp if home_gp > 0 else 1.1
        away_conceded_avg = away_total_conceded.get(away, 0) / away_gp if away_gp > 0 else 1.3
        
        # Thời điểm mùa giải
        try:
            matchday = int(g.get("round", idx % 38)) if pd.notna(g.get("round")) else idx % 38
        except:
            matchday = idx % 38
        
        is_start_season = 1 if matchday <= 5 else 0
        is_end_season = 1 if matchday >= 35 else 0
        
        # ============ TẠO ROW FEATURES ============
        
        row = {
            "date": match_date,
            "home_team": home,
            "away_team": away,
            
            # Elo
            "elo_diff": home_elo - away_elo,
            "elo_home": home_elo,
            "elo_away": away_elo,
            
            # Form
            "home_form": home_form,
            "away_form": away_form,
            "home_conceded": home_conceded,
            "away_conceded": away_conceded,
            "goal_diff_form": home_form - away_conceded - (away_form - home_conceded),
            
            # Form sân nhà/khách
            "home_form_home_games": home_form_home,
            "away_form_away_games": away_form_away,
            
            # H2H
            "h2h_diff": h2h_diff,
            "h2h_home_wins": h2h_home_wins,
            "h2h_away_wins": h2h_away_wins,
            "h2h_draws": h2h_draws,
            
            # Rest days
            "rest_days_diff": rest_days_diff,
            "home_rest_days": min(home_rest, 14),
            "away_rest_days": min(away_rest, 14),
            
            # Win rate
            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,
            "home_unbeaten_streak": min(home_streak, 15),
            "away_unbeaten_streak": min(away_streak, 15),
            
            # Points
            "home_points_per_game": home_ppg,
            "away_points_per_game": away_ppg,
            "points_diff": home_ppg - away_ppg,
            
            # Matchday
            "matchday": matchday,
            "is_start_season": is_start_season,
            "is_end_season": is_end_season,
            
            # Goals avg
            "home_goals_scored_avg": home_scored_avg,
            "away_goals_scored_avg": away_scored_avg,
            "home_goals_conceded_avg": home_conceded_avg,
            "away_goals_conceded_avg": away_conceded_avg,
            
            "target": g["target"],
        }
        
        rows.append(row)
        
        # ============ CẬP NHẬT STATE SAU TRẬN ĐẤU ============
        
        # Cập nhật form
        home_goals_for_hist.setdefault(home, []).append(g["home_score"])
        away_goals_for_hist.setdefault(away, []).append(g["away_score"])
        home_goals_against_hist.setdefault(home, []).append(g["away_score"])
        away_goals_against_hist.setdefault(away, []).append(g["home_score"])
        
        # Cập nhật form sân nhà/khách
        home_goals_home_games.setdefault(home, []).append(g["home_score"])
        away_goals_away_games.setdefault(away, []).append(g["away_score"])
        
        # Cập nhật H2H
        h2h_hist.setdefault(pair_key, []).append(
            (home, g["home_score"] - g["away_score"], match_date)
        )
        
        # Cập nhật last played
        last_played[home] = match_date
        last_played[away] = match_date
        
        # Cập nhật points
        home_pts, away_pts = _get_points(g["target"])
        home_points[home] = home_points.get(home, 0) + home_pts
        away_points[away] = away_points.get(away, 0) + away_pts
        home_games_played[home] = home_games_played.get(home, 0) + 1
        away_games_played[away] = away_games_played.get(away, 0) + 1
        
        # Cập nhật tổng bàn thắng/bại
        home_total_scored[home] = home_total_scored.get(home, 0) + g["home_score"]
        away_total_scored[away] = away_total_scored.get(away, 0) + g["away_score"]
        home_total_conceded[home] = home_total_conceded.get(home, 0) + g["away_score"]
        away_total_conceded[away] = away_total_conceded.get(away, 0) + g["home_score"]

        # Cập nhật lịch sử trận gần đây (cho biểu đồ phong độ W/D/L)
        home_ket_qua = "W" if g["target"] == 2 else ("D" if g["target"] == 1 else "L")
        away_ket_qua = "W" if g["target"] == 0 else ("D" if g["target"] == 1 else "L")
        recent_matches.setdefault(home, []).append({
            "date": match_date, "doi_thu": away, "san": "nha",
            "ban_thang": int(g["home_score"]), "ban_thua": int(g["away_score"]),
            "ket_qua": home_ket_qua,
        })
        recent_matches.setdefault(away, []).append({
            "date": match_date, "doi_thu": home, "san": "khach",
            "ban_thang": int(g["away_score"]), "ban_thua": int(g["home_score"]),
            "ket_qua": away_ket_qua,
        })
        
        # Cập nhật chuỗi bất bại
        if g["target"] == 2:  # Home win
            home_unbeaten[home] = home_unbeaten.get(home, 0) + 1
            away_unbeaten[away] = 0
        elif g["target"] == 0:  # Away win
            away_unbeaten[away] = away_unbeaten.get(away, 0) + 1
            home_unbeaten[home] = 0
        else:  # Draw
            home_unbeaten[home] = home_unbeaten.get(home, 0) + 1
            away_unbeaten[away] = away_unbeaten.get(away, 0) + 1
        
        # Cập nhật Elo
        expected_home = 1 / (1 + 10 ** (-((home_elo + ELO_HOME_ADVANTAGE) - away_elo) / 400))
        actual_home = 1.0 if g["target"] == 2 else (0.5 if g["target"] == 1 else 0.0)
        
        elo[home] = home_elo + ELO_K * (actual_home - expected_home)
        elo[away] = away_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))
    
    # Tạo DataFrame
    df = pd.DataFrame(rows)
    
    # Tạo team_state cuối cùng
    team_state = {
        "elo": elo,
        "home_form": {t: _safe_mean(v[-FORM_WINDOW:], 1.3) for t, v in home_goals_for_hist.items()},
        "away_form": {t: _safe_mean(v[-FORM_WINDOW:], 1.1) for t, v in away_goals_for_hist.items()},
        "home_conceded": {t: _safe_mean(v[-FORM_WINDOW:], 1.1) for t, v in home_goals_against_hist.items()},
        "away_conceded": {t: _safe_mean(v[-FORM_WINDOW:], 1.3) for t, v in away_goals_against_hist.items()},
        "home_form_home_games": {t: _safe_mean(v[-FORM_WINDOW:], 1.5) for t, v in home_goals_home_games.items()},
        "away_form_away_games": {t: _safe_mean(v[-FORM_WINDOW:], 1.0) for t, v in away_goals_away_games.items()},
        "h2h_hist": h2h_hist,
        "last_played": last_played,
        "home_points": home_points,
        "away_points": away_points,
        "home_games_played": home_games_played,
        "away_games_played": away_games_played,
        "home_unbeaten": home_unbeaten,
        "away_unbeaten": away_unbeaten,
        "home_total_scored": home_total_scored,
        "away_total_scored": away_total_scored,
        "home_total_conceded": home_total_conceded,
        "away_total_conceded": away_total_conceded,
        # Chỉ giữ 10 trận gần nhất/đội để tiết kiệm dung lượng cache.
        "recent_matches": {t: v[-10:] for t, v in recent_matches.items()},
    }
    
    logger.info(f"Xây dựng xong {len(df)} rows với {len(FEATURE_COLS)} features")
    
    return df, team_state


# ==================== MODEL TRAINING ====================

def _create_models() -> Dict[str, Any]:
    """Tạo dictionary các model candidates"""
    models = {}
    
    # Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # HistGradientBoosting
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42
    )
    
    # ExtraTrees
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # MLP Neural Network
    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    # XGBoost (nếu có)
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    
    # LightGBM (nếu có)
    if LGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    return models


def _evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict:
    """Đánh giá model trên test set"""
    try:
        # Fit và predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        # Tính metrics
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs, labels=[0, 1, 2])
        
        return {
            "accuracy": acc,
            "log_loss": ll,
            "predictions": preds,
            "probabilities": probs
        }
    except Exception as e:
        logger.error(f"Lỗi đánh giá model: {e}")
        return None


def train_model(league: str = LEAGUE, seasons: List[str] = SEASONS) -> Tuple[Any, Dict, List[str], Dict]:
    """
    Train model hoàn chỉnh với ensemble và feature selection
    
    Returns:
        model: Model đã train
        team_state: State cuối cùng của các đội
        teams: Danh sách các đội
        metrics: Metrics đánh giá
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAIN MODEL")
    logger.info("=" * 60)
    
    # Tải dữ liệu
    games = _load_raw_games(league, seasons)
    if len(games) == 0:
        raise ValueError("Không có dữ liệu để train")
    
    # Xây dựng features
    df, team_state = _build_features(games)

    # Bảng xếp hạng CHÍNH THỨC của mùa giải mới nhất (dùng cho tab "Bảng
    # xếp hạng" trên web) — tính riêng từ các trận của mùa đó, không lẫn
    # dữ liệu tích luỹ nhiều mùa như các key khác trong team_state.
    _, latest_season, season_games = _latest_season_slice(games, seasons)
    team_state["season_label"] = latest_season
    team_state["season_table"] = (
        _build_season_standings(season_games) if season_games is not None else []
    )
    
    # Chuẩn bị dữ liệu
    X = df[FEATURE_COLS].copy()
    y = df["target"].copy()
    
    # Xử lý missing values
    X = X.fillna(0)
    
    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Metrics
    metrics = {
        "so_tran_train": int(len(X_train)),
        "so_tran_test": int(len(X_test)),
        "accuracy": None,
        "log_loss": None,
        "model_type": None,
        "feature_count": len(FEATURE_COLS),
        "models_evaluated": {},
        "confusion_matrix": None
    }
    
    if len(X_test) < 20:
        logger.warning("Không đủ dữ liệu test, bỏ qua đánh giá")
        metrics["model_type"] = "random_forest"
        final_model = CalibratedClassifierCV(
            RandomForestClassifier(random_state=42),
            method="isotonic",
            cv=5
        )
        final_model.fit(X, y)
    else:
        # ============ FEATURE SELECTION ============
        logger.info("Đang thực hiện feature selection...")
        
        selector = SelectFromModel(
            ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            max_features=min(20, len(FEATURE_COLS)),
            threshold=-np.inf
        )
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Lấy feature names được chọn
        selected_features_mask = selector.get_support()
        selected_features = [f for f, m in zip(FEATURE_COLS, selected_features_mask) if m]
        
        logger.info(f"Chọn {len(selected_features)}/{len(FEATURE_COLS)} features quan trọng nhất")
        logger.info(f"Features được chọn: {selected_features[:10]}...")
        
        # ============ SCALING CHO MLP ============
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # ============ TRAIN VÀ ĐÁNH GIÁ CÁC MODEL ============
        logger.info("Đang train và đánh giá các model...")
        
        models = _create_models()
        best_model = None
        best_score = float('inf')
        model_results = {}
        
        for name, base_model in models.items():
            logger.info(f"Đánh giá model: {name}...")
            
            try:
                # Calibrated model
                calibrated = CalibratedClassifierCV(
                    base_model,
                    method="isotonic",
                    cv=5
                )
                
                # Dùng scaled data cho MLP
                if name == "mlp":
                    X_tr = X_train_scaled
                    X_te = X_test_scaled
                else:
                    X_tr = X_train_selected
                    X_te = X_test_selected
                
                result = _evaluate_model(calibrated, X_tr, y_train, X_te, y_test)
                
                if result:
                    model_results[name] = {
                        "accuracy": round(result["accuracy"] * 100, 2),
                        "log_loss": round(result["log_loss"], 4)
                    }
                    
                    logger.info(f"  {name}: accuracy={result['accuracy']*100:.2f}%, "
                              f"log_loss={result['log_loss']:.4f}")
                    
                    if result["log_loss"] < best_score:
                        best_score = result["log_loss"]
                        best_model = calibrated
                        metrics["accuracy"] = round(result["accuracy"] * 100, 2)
                        metrics["log_loss"] = round(result["log_loss"], 4)
                        metrics["model_type"] = name
                        
                        # Confusion matrix
                        metrics["confusion_matrix"] = confusion_matrix(
                            y_test, result["predictions"]
                        ).tolist()
            except Exception as e:
                logger.error(f"Lỗi với model {name}: {e}")
                continue
        
        metrics["models_evaluated"] = model_results
        
        # ============ ENSEMBLE ============
        logger.info("Đang tạo ensemble model...")
        
        # Lấy top 3 models tốt nhất
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]["log_loss"])
        top_models = [(name, models[name]) for name, _ in sorted_models[:3]]
        
        if len(top_models) >= 2:
            logger.info(f"Ensemble với {len(top_models)} models tốt nhất")
            
            # Voting ensemble
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in top_models],
                voting='soft'
            )
            
            # Calibrate ensemble
            ensemble_calibrated = CalibratedClassifierCV(
                ensemble,
                method="isotonic",
                cv=5
            )
            
            # Đánh giá ensemble
            if "mlp" in [name for name, _ in top_models]:
                ensemble_result = _evaluate_model(
                    ensemble_calibrated, X_train_scaled, y_train, X_test_scaled, y_test
                )
            else:
                ensemble_result = _evaluate_model(
                    ensemble_calibrated, X_train_selected, y_train, X_test_selected, y_test
                )
            
            if ensemble_result and ensemble_result["log_loss"] < best_score:
                logger.info(f"Ensemble tốt hơn! log_loss={ensemble_result['log_loss']:.4f}")
                best_model = ensemble_calibrated
                best_score = ensemble_result["log_loss"]
                metrics["accuracy"] = round(ensemble_result["accuracy"] * 100, 2)
                metrics["log_loss"] = round(ensemble_result["log_loss"], 4)
                metrics["model_type"] = "ensemble"
        
        # ============ TRAIN FINAL MODEL TRÊN TOÀN BỘ DATA ============
        logger.info("Train final model trên toàn bộ data...")
        
        if metrics["model_type"] == "ensemble":
            final_model = VotingClassifier(
                estimators=[(name, models[name]) for name, _ in top_models],
                voting='soft'
            )
        else:
            final_model = models.get(metrics["model_type"], models["random_forest"])
        
        # Calibrate final model
        final_model = CalibratedClassifierCV(
            final_model,
            method="isotonic",
            cv=5
        )
        
        # Fit trên toàn bộ data
        if metrics["model_type"] == "mlp":
            X_all = scaler.fit_transform(selector.transform(X))
        else:
            X_all = selector.transform(X)
        
        final_model.fit(X_all, y)
        
        # Lưu scaler và selector vào model
        final_model.scaler_ = scaler
        final_model.selector_ = selector
        final_model.selected_features_ = selected_features
        final_model.is_mlp_ = metrics["model_type"] == "mlp"
    
   
    # Lấy danh sách teams — chỉ những đội đang đá mùa MỚI NHẤT, không lẫn
    # đội cũ đã xuống hạng từ các mùa trước trong SEASONS.
    teams = _get_current_season_teams(games, seasons)
    
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH TRAIN MODEL")
    logger.info(f"Model: {metrics['model_type']}")
    logger.info(f"Accuracy: {metrics['accuracy']}%")
    logger.info(f"Log loss: {metrics['log_loss']}")
    logger.info("=" * 60)
    
    return final_model, team_state, teams, metrics


# ==================== CACHE MANAGEMENT ====================

def load_or_train_model(
    force_retrain: bool = False,
    league_key: str = DEFAULT_LEAGUE_KEY,
) -> Tuple[Any, Dict, List[str], Dict]:
    """
    Load model từ cache hoặc train mới nếu cache cũ.
    league_key: 1 trong các key của LEAGUES (vd "la-liga"). Mỗi giải có
    file cache riêng nên train/giữ nhiều giải cùng lúc không đè lên nhau.
    """
    if league_key not in LEAGUES:
        raise ValueError(
            f'Giải đấu "{league_key}" chưa được hỗ trợ. '
            f"Các giải hiện có: {', '.join(LEAGUES.keys())}"
        )

    cache_path = _cache_path(league_key)

    if not force_retrain and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            
            age = time.time() - cached.get("trained_at", 0)
            if age < CACHE_MAX_AGE_SECONDS:
                logger.info(f"Dùng model cache của {league_key} (train cách đây {age/3600:.1f} giờ)")
                return cached["model"], cached["team_state"], cached["teams"], cached["metrics"]
            else:
                logger.info(f"Cache {league_key} cũ ({age/3600:.1f} giờ), train lại...")
        except Exception as e:
            logger.warning(f"Lỗi đọc cache {league_key}: {e}, train lại...")
    
    # Train model mới
    league_cfg = LEAGUES[league_key]
    model, team_state, teams, metrics = train_model(
        league=league_cfg["code"], seasons=league_cfg["seasons"]
    )
    
    # Lưu cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "model": model,
                "team_state": team_state,
                "teams": teams,
                "metrics": metrics,
                "trained_at": time.time(),
                "version": "2.0",
                "league_key": league_key,
            }, f)
        logger.info(f"Đã lưu model cache mới cho {league_key}")
    except Exception as e:
        logger.warning(f"Không thể lưu cache {league_key}: {e}")
    
    return model, team_state, teams, metrics


# ==================== PREDICTION ====================

def predict_match(model, team_state: Dict, doi_nha: str, doi_khach: str) -> Dict:
    """
    Dự đoán tỉ lệ thắng/hòa/thua cho 1 trận đấu sắp diễn ra, dùng toàn bộ
    feature đã xây dựng khi train (Elo, form, H2H, rest days, win rate,
    points per game, matchday...), tính từ team_state mới nhất.

    Model cuối cùng (final_model) có thể là 1 model đơn hoặc ensemble, và
    luôn đi kèm 3 thuộc tính được gắn thêm lúc train:
      - model.selector_        : SelectFromModel đã fit, dùng để chọn ra
                                  đúng tập feature quan trọng nhất.
      - model.scaler_          : StandardScaler (chỉ áp dụng khi model
                                  cuối cùng là MLP, vì MLP nhạy với scale).
      - model.is_mlp_          : True nếu model_type cuối cùng là "mlp".

    Trả về dict kết quả, hoặc raise ValueError nếu tên đội không hợp lệ.
    """
    elo = team_state["elo"]
    if doi_nha not in elo:
        raise ValueError(f'Không tìm thấy đội nhà "{doi_nha}" trong dữ liệu.')
    if doi_khach not in elo:
        raise ValueError(f'Không tìm thấy đội khách "{doi_khach}" trong dữ liệu.')

    # ============ ELO ============
    home_elo = elo[doi_nha]
    away_elo = elo[doi_khach]

    # ============ FORM CHUNG (5 trận gần nhất) ============
    home_form = team_state["home_form"].get(doi_nha, 1.3)
    away_form = team_state["away_form"].get(doi_khach, 1.1)
    home_conceded = team_state["home_conceded"].get(doi_nha, 1.1)
    away_conceded = team_state["away_conceded"].get(doi_khach, 1.3)

    # ============ FORM RIÊNG SÂN NHÀ/SÂN KHÁCH ============
    home_form_home = team_state["home_form_home_games"].get(doi_nha, 1.5)
    away_form_away = team_state["away_form_away_games"].get(doi_khach, 1.0)

    # ============ HEAD-TO-HEAD ============
    # h2h_hist lưu tuple (đội_nhà_lần_đó, hiệu_số_bàn_thắng, ngày_đá) — định
    # hướng lại hiệu số theo góc nhìn "doi_nha là đội nhà" ở trận sắp tới.
    pair_key = frozenset((doi_nha, doi_khach))
    past_meetings = team_state.get("h2h_hist", {}).get(pair_key, [])[-H2H_WINDOW:]
    if past_meetings:
        oriented_diffs = [
            diff if past_home == doi_nha else -diff
            for past_home, diff, _ in past_meetings
        ]
        h2h_diff = float(np.mean(oriented_diffs))
    else:
        h2h_diff = 0.0

    h2h_home_wins = h2h_away_wins = h2h_draws = 0
    for past_home, diff, _ in past_meetings:
        oriented = diff if past_home == doi_nha else -diff
        if oriented > 0:
            h2h_home_wins += 1
        elif oriented < 0:
            h2h_away_wins += 1
        else:
            h2h_draws += 1

    # ============ SỐ NGÀY NGHỈ ============
    # Dùng ngày hiện tại (lúc gọi API) so với lần đá gần nhất của mỗi đội —
    # đây là ước tính hợp lý cho 1 trận sắp diễn ra (khác với lúc train, khi
    # ta biết chính xác ngày đá thật của từng trận lịch sử).
    last_played = team_state.get("last_played", {})
    now = datetime.now()
    home_last = last_played.get(doi_nha)
    away_last = last_played.get(doi_khach)
    home_rest = (now - home_last).days if home_last is not None else 7
    away_rest = (now - away_last).days if away_last is not None else 7
    rest_days_diff = float(np.clip(home_rest - away_rest, -14, 14))
    home_rest_days = float(min(max(home_rest, 0), 14))
    away_rest_days = float(min(max(away_rest, 0), 14))

    # ============ WIN RATE & POINTS PER GAME ============
    home_gp = team_state.get("home_games_played", {}).get(doi_nha, 0)
    away_gp = team_state.get("away_games_played", {}).get(doi_khach, 0)
    home_points = team_state.get("home_points", {}).get(doi_nha, 0)
    away_points = team_state.get("away_points", {}).get(doi_khach, 0)

    home_win_rate = home_points / (home_gp * 3) if home_gp > 0 else 0.35
    away_win_rate = away_points / (away_gp * 3) if away_gp > 0 else 0.30
    home_ppg = home_points / home_gp if home_gp > 0 else 1.3
    away_ppg = away_points / away_gp if away_gp > 0 else 1.1
    points_diff = home_ppg - away_ppg

    # ============ CHUỖI BẤT BẠI ============
    home_streak = min(team_state.get("home_unbeaten", {}).get(doi_nha, 0), 15)
    away_streak = min(team_state.get("away_unbeaten", {}).get(doi_khach, 0), 15)

    # ============ TỶ LỆ GHI BÀN TRUNG BÌNH ============
    home_total_scored = team_state.get("home_total_scored", {}).get(doi_nha, 0)
    away_total_scored = team_state.get("away_total_scored", {}).get(doi_khach, 0)
    home_total_conceded = team_state.get("home_total_conceded", {}).get(doi_nha, 0)
    away_total_conceded = team_state.get("away_total_conceded", {}).get(doi_khach, 0)

    home_scored_avg = home_total_scored / home_gp if home_gp > 0 else 1.3
    away_scored_avg = away_total_scored / away_gp if away_gp > 0 else 1.1
    home_conceded_avg = home_total_conceded / home_gp if home_gp > 0 else 1.1
    away_conceded_avg = away_total_conceded / away_gp if away_gp > 0 else 1.3

    # ============ THỜI ĐIỂM MÙA GIẢI (ước tính) ============
    # Trận sắp đá chưa có "round" thật, nên ước tính bằng số trận trung bình
    # 2 đội đã đá — không chính xác 100% nhưng hợp lý hơn nhiều so với việc
    # bỏ trống hoàn toàn hoặc luôn coi là đầu/cuối mùa.
    matchday = int(round((home_gp + away_gp) / 2)) + 1
    matchday = max(1, min(matchday, 38))
    is_start_season = 1 if matchday <= 5 else 0
    is_end_season = 1 if matchday >= 35 else 0

    # ============ GHÉP FEATURE ĐÚNG THỨ TỰ FEATURE_COLS ============
    raw_features = pd.DataFrame([{
        "elo_diff": home_elo - away_elo,
        "elo_home": home_elo,
        "elo_away": away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
        "goal_diff_form": home_form - away_conceded - (away_form - home_conceded),
        "home_form_home_games": home_form_home,
        "away_form_away_games": away_form_away,
        "h2h_diff": h2h_diff,
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
        "rest_days_diff": rest_days_diff,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_win_rate": home_win_rate,
        "away_win_rate": away_win_rate,
        "home_unbeaten_streak": home_streak,
        "away_unbeaten_streak": away_streak,
        "home_points_per_game": home_ppg,
        "away_points_per_game": away_ppg,
        "points_diff": points_diff,
        "matchday": matchday,
        "is_start_season": is_start_season,
        "is_end_season": is_end_season,
        "home_goals_scored_avg": home_scored_avg,
        "away_goals_scored_avg": away_scored_avg,
        "home_goals_conceded_avg": home_conceded_avg,
        "away_goals_conceded_avg": away_conceded_avg,
    }])[FEATURE_COLS].fillna(0)

    # ============ ÁP DỤNG FEATURE SELECTION (+ SCALING NẾU LÀ MLP) ============
    # Phải dùng ĐÚNG selector/scaler đã fit lúc train (gắn sẵn vào model),
    # nếu không feature sẽ lệch cột và model dự đoán sai/lỗi.
    selector = getattr(model, "selector_", None)
    scaler = getattr(model, "scaler_", None)
    is_mlp = getattr(model, "is_mlp_", False)

    if selector is not None:
        features = selector.transform(raw_features)
    else:
        features = raw_features.values

    if is_mlp and scaler is not None:
        features = scaler.transform(features)

    # ============ DỰ ĐOÁN ============
    probs = model.predict_proba(features)[0]
    # Thứ tự class của model: 0=khách thắng, 1=hoà, 2=nhà thắng (theo _get_result)
    class_order = list(model.classes_)

    def _p(cls):
        return round(float(probs[class_order.index(cls)]) * 100, 1)

    # ============ GIẢI THÍCH NGẮN (hiển thị cho mọi người dùng) ============
    # Không cần model giải thích được (Random Forest không có "lý do" tường
    # minh) — chỉ cần liệt lại vài chênh lệch số liệu lớn nhất theo thứ tự
    # dễ hiểu, người xem tự suy ra vì sao tỉ lệ nghiêng về bên nào.
    explain = []

    elo_diff_abs = home_elo - away_elo
    if abs(elo_diff_abs) >= 15:
        leader = doi_nha if elo_diff_abs > 0 else doi_khach
        explain.append(f"{leader} có điểm Elo cao hơn ({round(abs(elo_diff_abs))} điểm).")

    if abs(points_diff) >= 0.3:
        leader = doi_nha if points_diff > 0 else doi_khach
        explain.append(f"{leader} đang có phong độ điểm số tốt hơn (điểm/trận cao hơn).")

    if past_meetings:
        if h2h_diff > 0.3:
            explain.append(f"{doi_nha} thường thắng đậm hơn trong các lần đối đầu gần đây.")
        elif h2h_diff < -0.3:
            explain.append(f"{doi_khach} thường thắng đậm hơn trong các lần đối đầu gần đây.")
        else:
            explain.append("Lịch sử đối đầu gần đây khá cân bằng giữa hai đội.")
    else:
        explain.append("Chưa có dữ liệu đối đầu gần đây giữa hai đội.")

    if home_streak >= 4:
        explain.append(f"{doi_nha} đang bất bại {home_streak} trận sân nhà gần nhất.")
    if away_streak >= 4:
        explain.append(f"{doi_khach} đang bất bại {away_streak} trận sân khách gần nhất.")

    if abs(rest_days_diff) >= 3:
        rested = doi_nha if rest_days_diff < 0 else doi_khach
        explain.append(f"{rested} có nhiều ngày nghỉ hơn trước trận này.")

    if not explain:
        explain.append("Hai đội khá cân bằng về các chỉ số gần đây, tỉ lệ nghiêng nhẹ theo dữ liệu lịch sử tổng thể.")

    # ============ DỰ ĐOÁN TỶ SỐ CHÍNH XÁC (Poisson) ============
    # Ước lượng số bàn kỳ vọng mỗi đội từ TB bàn ghi được của đội này và TB
    # bàn để thủng của đối thủ (cách làm phổ biến, không cần model riêng),
    # sau đó dựng phân phối Poisson độc lập cho mỗi bên và ghép lại thành
    # ma trận xác suất của từng tỷ số cụ thể.
    lambda_home = max(0.25, (home_scored_avg + away_conceded_avg) / 2)
    lambda_away = max(0.25, (away_scored_avg + home_conceded_avg) / 2)

    def _poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    MAX_GOALS = 6
    score_grid = []
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = _poisson_pmf(hg, lambda_home) * _poisson_pmf(ag, lambda_away)
            score_grid.append((hg, ag, p))
    total_p = sum(p for _, _, p in score_grid) or 1.0
    score_grid = [(hg, ag, p / total_p) for hg, ag, p in score_grid]
    score_grid.sort(key=lambda x: x[2], reverse=True)
    top_scores = [
        {"ty_so": f"{hg}-{ag}", "xac_suat": round(p * 100, 1)}
        for hg, ag, p in score_grid[:5]
    ]
    ty_so_chinh_xac = {
        "du_doan_nhat": top_scores[0]["ty_so"] if top_scores else None,
        "top5": top_scores,
        "ban_thang_ky_vong": {"nha": round(lambda_home, 2), "khach": round(lambda_away, 2)},
    }

    # ============ BIỂU ĐỒ PHONG ĐỘ (5 trận gần nhất, W/D/L) ============
    recent_matches = team_state.get("recent_matches", {})

    def _form_chart(matches):
        return [
            {
                "ket_qua": m["ket_qua"],
                "doi_thu": m["doi_thu"],
                "ty_so": f'{m["ban_thang"]}-{m["ban_thua"]}',
                "san": m["san"],
            }
            for m in matches[-5:]
        ]

    bieu_do_phong_do = {
        "nha": _form_chart(recent_matches.get(doi_nha, [])),
        "khach": _form_chart(recent_matches.get(doi_khach, [])),
    }

    # ============ SO SÁNH HAI ĐỘI (miễn phí, để thu hút người dùng) ========
    so_sanh = {
        "elo": {"home": round(home_elo), "away": round(away_elo)},
        "phong_do_pct": {
            "home": round(min(100, (home_form / 3) * 100)),
            "away": round(min(100, (away_form / 3) * 100)),
        },
        "ban_thang_tb": {"home": round(home_scored_avg, 2), "away": round(away_scored_avg, 2)},
        "ban_thua_tb": {"home": round(home_conceded_avg, 2), "away": round(away_conceded_avg, 2)},
        "win_rate_pct": {
            "home": round(home_win_rate * 100),
            "away": round(away_win_rate * 100),
        },
    }

    # ============ ĐỘ TIN CẬY CỦA AI (KHÔNG PHẢI XÁC SUẤT THẮNG) ==========
    probs_pct = sorted([_p(2), _p(1), _p(0)], reverse=True)
    gap = probs_pct[0] - probs_pct[1]  # khoảng cách giữa lựa chọn khả dĩ nhất và lựa chọn còn lại
    data_volume = min(home_gp, away_gp)

    gap_component = min(100, gap * 2.2)
    elo_component = min(100, abs(elo_diff_abs) / 2)
    data_component = min(100, (data_volume / 25) * 100)
    confidence_score = round(0.45 * gap_component + 0.30 * elo_component + 0.25 * data_component)
    confidence_score = int(max(5, min(96, confidence_score)))

    if confidence_score >= 70:
        confidence_level = "CAO"
    elif confidence_score >= 45:
        confidence_level = "TRUNG BÌNH"
    else:
        confidence_level = "THẤP"

    confidence_reasons = []
    if abs(elo_diff_abs) >= 50:
        confidence_reasons.append("Chênh lệch Elo giữa hai đội khá lớn.")
    if gap >= 25:
        confidence_reasons.append("Kết quả khả dĩ nhất bỏ xa các phương án còn lại.")
    if data_volume >= 20:
        confidence_reasons.append("Đủ dữ liệu lịch sử để mô hình học ổn định.")
    elif data_volume < 8:
        confidence_reasons.append("Dữ liệu của ít nhất 1 đội còn khá mỏng, độ tin cậy có thể giảm.")
    if past_meetings:
        confidence_reasons.append("Có lịch sử đối đầu trực tiếp giữa hai đội để tham khảo.")
    if not confidence_reasons:
        confidence_reasons.append("Các chỉ số của hai đội khá cân bằng, nên độ chắc chắn ở mức vừa phải.")

    do_tin_cay = {
        "diem": confidence_score,
        "muc": confidence_level,
        "ly_do": confidence_reasons,
        "ghi_chu": "Đây là độ tự tin của mô hình dựa trên chất lượng & sự khác biệt dữ liệu, không phải xác suất thắng chắc chắn.",
    }

    # ============ THỐNG KÊ CHI TIẾT (chỉ hiện với tài khoản Premium) ========
    premium_stats = {
        "elo": {"home": round(home_elo), "away": round(away_elo)},
        "form_5_tran": {"home": round(home_form, 2), "away": round(away_form, 2)},
        "diem_moi_tran": {"home": round(home_ppg, 2), "away": round(away_ppg, 2)},
        "doi_dau_gan_day": {
            "so_tran": len(past_meetings),
            "home_thang": h2h_home_wins,
            "hoa": h2h_draws,
            "away_thang": h2h_away_wins,
        },
        "ngay_nghi": {"home": round(home_rest_days), "away": round(away_rest_days)},
        "bat_bai_lien_tiep": {"home": home_streak, "away": away_streak},
        "ban_thang_tb": {"home": round(home_scored_avg, 2), "away": round(away_scored_avg, 2)},
        "ban_thua_tb": {"home": round(home_conceded_avg, 2), "away": round(away_conceded_avg, 2)},
    }

    return {
        "doi_nha": doi_nha,
        "doi_khach": doi_khach,
        "thang_nha": _p(2),
        "hoa": _p(1),
        "thang_khach": _p(0),
        "explain": explain,
        "premium_stats": premium_stats,
        "ty_so_chinh_xac": ty_so_chinh_xac,
        "bieu_do_phong_do": bieu_do_phong_do,
        "so_sanh": so_sanh,
        "do_tin_cay": do_tin_cay,
    }


# ==================== BẢNG XẾP HẠNG ====================

# key -> nhãn hiển thị cho FE (dropdown chọn thông số sắp xếp).
# Thứ tự dict cũng là thứ tự hiện trong dropdown.
LEADERBOARD_SORT_FIELDS = {
    "diem": "Điểm",
    "diem_moi_tran": "Điểm / trận",
    "elo": "Điểm Elo",
    "hieu_so": "Hiệu số bàn thắng",
    "ban_thang": "Bàn thắng",
    "ban_thua": "Bàn thua",
    "phong_do_ghi_ban": "Phong độ ghi bàn (TB 5 trận gần nhất)",
    "bat_bai_lien_tiep": "Chuỗi bất bại",
}

# Các thông số càng THẤP càng tốt -> sắp xếp tăng dần thay vì giảm dần.
_LEADERBOARD_ASCENDING_FIELDS = {"ban_thua"}


def build_leaderboard(team_state: Dict, teams: List[str]) -> List[Dict]:
    """
    Xây dựng bảng xếp hạng các đội trong 1 giải đấu, từ các chỉ số đã tích
    luỹ trong team_state (điểm, bàn thắng/bại, Elo, phong độ...).

    LƯU Ý QUAN TRỌNG: team_state tích luỹ dữ liệu từ NHIỀU MÙA GIẢI (theo
    LEAGUES[...]["seasons"]), không tách riêng theo từng mùa -> đây KHÔNG
    phải bảng xếp hạng chính thức của 1 mùa giải cụ thể (như trên
    Premier League/FBref), mà là bảng xếp hạng "sức mạnh tổng thể" dựa
    trên toàn bộ dữ liệu mô hình đã học được. Elo và điểm/trận vẫn phản
    ánh khá tốt phong độ tương đối giữa các đội đang thi đấu mùa hiện tại.
    """
    elo = team_state.get("elo", {})
    home_points = team_state.get("home_points", {})
    away_points = team_state.get("away_points", {})
    home_gp = team_state.get("home_games_played", {})
    away_gp = team_state.get("away_games_played", {})
    home_scored = team_state.get("home_total_scored", {})
    away_scored = team_state.get("away_total_scored", {})
    home_conceded = team_state.get("home_total_conceded", {})
    away_conceded = team_state.get("away_total_conceded", {})
    home_form = team_state.get("home_form", {})
    away_form = team_state.get("away_form", {})
    home_unbeaten = team_state.get("home_unbeaten", {})
    away_unbeaten = team_state.get("away_unbeaten", {})

    rows = []
    for team in teams:
        so_tran = home_gp.get(team, 0) + away_gp.get(team, 0)
        diem = home_points.get(team, 0) + away_points.get(team, 0)
        ban_thang = home_scored.get(team, 0) + away_scored.get(team, 0)
        ban_thua = home_conceded.get(team, 0) + away_conceded.get(team, 0)

        form_vals = [v for v in (home_form.get(team), away_form.get(team)) if v is not None]
        phong_do = float(np.mean(form_vals)) if form_vals else 0.0

        rows.append({
            "doi": team,
            "elo": round(elo.get(team, ELO_START)),
            "so_tran": so_tran,
            "diem": round(diem, 1),
            "diem_moi_tran": round(diem / so_tran, 2) if so_tran > 0 else 0.0,
            "ban_thang": ban_thang,
            "ban_thua": ban_thua,
            "hieu_so": ban_thang - ban_thua,
            "phong_do_ghi_ban": round(phong_do, 2),
            "bat_bai_lien_tiep": max(
                home_unbeaten.get(team, 0), away_unbeaten.get(team, 0)
            ),
        })

    return rows


def sort_leaderboard(rows: List[Dict], sort_by: str = "diem") -> List[Dict]:
    """Sắp xếp bảng xếp hạng theo 1 thông số và gán số thứ hạng ("hang").
    sort_by không hợp lệ -> tự động dùng "diem" (điểm số) làm mặc định."""
    key = sort_by if sort_by in LEADERBOARD_SORT_FIELDS else "diem"
    sorted_rows = sorted(
        rows,
        key=lambda r: r.get(key, 0),
        reverse=key not in _LEADERBOARD_ASCENDING_FIELDS,
    )
    for i, r in enumerate(sorted_rows, start=1):
        r["hang"] = i
    return sorted_rows