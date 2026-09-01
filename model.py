"""
model.py
--------
Chứa toàn bộ logic AI: tải dữ liệu, huấn luyện mô hình, và dự đoán.

Phiên bản nâng cấp so với bản gốc — những gì đã thay đổi và TẠI SAO:

1. HẾT DATA LEAKAGE.
   Bản gốc tính "phong độ" của mỗi đội bằng trung bình bàn thắng CẢ MÙA,
   tức là khi dự đoán/huấn luyện cho một trận, model đã "nhìn thấy" luôn
   kết quả của các trận đá SAU trận đó. Model học kiểu này trông có vẻ giỏi
   khi test ngược lại chính dữ liệu train, nhưng vô dụng khi dùng dự đoán
   trận thật (chưa đá). Bản này duyệt trận đấu theo đúng thứ tự thời gian
   và chỉ dùng dữ liệu QUÁ KHỨ để tính feature cho mỗi trận.

2. NHIỀU FEATURE HƠN, PHẢN ÁNH SỨC MẠNH ĐỘI BÓNG TỐT HƠN.
   - Elo rating: cập nhật dần sau mỗi trận (thắng đội mạnh tăng nhiều điểm
     hơn thắng đội yếu), có cộng thêm lợi thế sân nhà.
   - Phong độ ghi bàn/thủng lưới 5 trận gần nhất (thay vì trung bình cả mùa
     "loãng" thông tin của phong độ hiện tại).
   - Hiệu số phong độ tấn công/phòng ngự giữa hai đội.

3. ĐÁNH GIÁ MODEL THẬT SỰ.
   Tách 20% trận gần nhất ra làm tập test theo đúng THỜI GIAN (không xáo
   trộn ngẫu nhiên — xáo trộn cũng là một dạng leakage), rồi đo accuracy và
   log loss trên đó. Số liệu này được trả về để app.py hiển thị cho người
   dùng biết model đáng tin tới đâu.

4. XÁC SUẤT ĐÁNG TIN HƠN.
   RandomForest thô thường cho xác suất bị lệch (ví dụ hay đoán quá tự tin).
   Dùng CalibratedClassifierCV để hiệu chỉnh lại, % trả về sát với tần suất
   thắng/thua thật hơn.

5. CACHE CÓ HẠN SỬ DỤNG.
   Cache cũ giờ có timestamp, tự động train lại nếu quá cũ thay vì dùng mãi
   một cache lỗi thời.
"""

import os
import pickle
import time
import logging
import datetime
import numpy as np
import pandas as pd
import soccerdata as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "model_cache.pkl")

LEAGUE = "ENG-Premier League"


def _current_season_start_year(today: datetime.date = None) -> int:
    """
    Premier League khởi tranh khoảng tháng 8 và kết thúc khoảng tháng 5 năm
    sau. Coi mùa giải "hiện tại" bắt đầu từ tháng 7 (có transfer window hè)
    để mã mùa luôn nhảy sang mùa mới đúng lúc, kể cả trước khi trận đầu mùa
    diễn ra.
    """
    today = today or datetime.date.today()
    return today.year if today.month >= 7 else today.year - 1


def _season_code(start_year: int) -> str:
    """VD: 2026 -> '2627' (mùa 2026-27), khớp định dạng soccerdata dùng."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def _recent_seasons(n_seasons: int = 3, today: datetime.date = None):
    """
    Tự tính ra n_seasons mùa gần nhất, LUÔN bao gồm mùa đang diễn ra —
    không cần sửa tay mỗi khi sang mùa giải mới như bản trước.
    """
    start = _current_season_start_year(today)
    return [_season_code(start - i) for i in range(n_seasons - 1, -1, -1)]


# Tự động luôn lấy 3 mùa gần nhất tính đến ngày chạy code, bao gồm mùa hiện tại.
SEASONS = _recent_seasons(3)

# Cache được coi là "cũ" sau bao nhiêu giây thì tự train lại.
# Rút xuống 1 ngày (thay vì 3) vì đầu mùa giải phong độ đội bóng thay đổi
# nhanh (chuyển nhượng, đổi HLV...), cache cũ dễ lệch thực tế hơn bình thường.
CACHE_MAX_AGE_SECONDS = 1 * 24 * 60 * 60

# Hệ số Elo
ELO_K = 20
ELO_HOME_ADVANTAGE = 60  # điểm Elo cộng thêm cho đội đá sân nhà
ELO_START = 1500

# Số trận gần nhất dùng để tính "phong độ" (rolling form)
FORM_WINDOW = 5

FEATURE_COLS = ["elo_diff", "home_form", "away_form", "home_conceded", "away_conceded", "goal_diff_form"]


def _get_result(row):
    """2 = đội nhà thắng, 1 = hòa, 0 = đội khách thắng"""
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] == row["away_score"]:
        return 1
    return 0


def _load_raw_games(league: str, seasons):
    fbref = sd.FBref(leagues=league, seasons=seasons)
    games = fbref.read_schedule().reset_index()

    completed = games[games["game_id"].notnull()].copy()
    completed[["home_score", "away_score"]] = (
        completed["score"].str.split("–|-", regex=True, expand=True).astype(float)
    )
    completed["target"] = completed.apply(_get_result, axis=1)

    # Sắp xếp đúng thứ tự thời gian — bắt buộc để tính feature không bị leak
    completed = completed.sort_values("date").reset_index(drop=True)
    return completed


def _build_features(games: pd.DataFrame):
    """
    Duyệt qua từng trận theo đúng thứ tự thời gian, và với MỖI trận:
      - đọc feature hiện tại của 2 đội (chỉ từ quá khứ)
      - rồi mới cập nhật feature bằng kết quả trận đó

    => Không có trận nào "nhìn thấy" kết quả của chính nó hay trận tương lai.
    """
    elo = {}
    home_goals_for_hist, away_goals_for_hist = {}, {}
    home_goals_against_hist, away_goals_against_hist = {}, {}

    rows = []

    for _, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]

        home_elo = elo.get(home, ELO_START)
        away_elo = elo.get(away, ELO_START)

        home_form = np.mean(home_goals_for_hist.get(home, [])[-FORM_WINDOW:]) if home_goals_for_hist.get(home) else 1.3
        away_form = np.mean(away_goals_for_hist.get(away, [])[-FORM_WINDOW:]) if away_goals_for_hist.get(away) else 1.1
        home_conceded = np.mean(home_goals_against_hist.get(home, [])[-FORM_WINDOW:]) if home_goals_against_hist.get(home) else 1.1
        away_conceded = np.mean(away_goals_against_hist.get(away, [])[-FORM_WINDOW:]) if away_goals_against_hist.get(away) else 1.3

        rows.append({
            "date": g["date"],
            "home_team": home,
            "away_team": away,
            "elo_diff": home_elo - away_elo,
            "home_form": home_form,
            "away_form": away_form,
            "home_conceded": home_conceded,
            "away_conceded": away_conceded,
            "goal_diff_form": home_form - away_conceded - (away_form - home_conceded),
            "target": g["target"],
        })

        # --- cập nhật state SAU KHI đã lấy feature cho trận này ---
        home_goals_for_hist.setdefault(home, []).append(g["home_score"])
        away_goals_for_hist.setdefault(away, []).append(g["away_score"])
        home_goals_against_hist.setdefault(home, []).append(g["away_score"])
        away_goals_against_hist.setdefault(away, []).append(g["home_score"])

        expected_home = 1 / (1 + 10 ** (-((home_elo + ELO_HOME_ADVANTAGE) - away_elo) / 400))
        actual_home = 1.0 if g["target"] == 2 else (0.5 if g["target"] == 1 else 0.0)
        elo[home] = home_elo + ELO_K * (actual_home - expected_home)
        elo[away] = away_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))

    df = pd.DataFrame(rows)

    team_state = {
        "elo": elo,
        "home_form": {t: (np.mean(v[-FORM_WINDOW:]) if v else 1.3) for t, v in home_goals_for_hist.items()},
        "away_form": {t: (np.mean(v[-FORM_WINDOW:]) if v else 1.1) for t, v in away_goals_for_hist.items()},
        "home_conceded": {t: (np.mean(v[-FORM_WINDOW:]) if v else 1.1) for t, v in home_goals_against_hist.items()},
        "away_conceded": {t: (np.mean(v[-FORM_WINDOW:]) if v else 1.3) for t, v in away_goals_against_hist.items()},
    }

    return df, team_state


def train_model(league: str = LEAGUE, seasons=SEASONS):
    """
    Tải dữ liệu lịch sử từ FBref, xây feature không bị leak, huấn luyện lại
    mô hình từ đầu, và đánh giá trên tập test theo thời gian.

    Trả về (model, team_state, teams, metrics):
      - team_state: Elo/form mới nhất của từng đội, dùng khi predict.
      - metrics: dict {"so_tran_train", "so_tran_test", "accuracy", "log_loss"}
        (accuracy/log_loss = None nếu không đủ dữ liệu để đánh giá).
    """
    logger.info("Đang tải dữ liệu từ FBref cho %s mùa %s...", league, seasons)
    games = _load_raw_games(league, seasons)
    logger.info("Tải xong %d trận đã đá.", len(games))

    df, team_state = _build_features(games)

    X = df[FEATURE_COLS]
    y = df["target"]

    # Time-based split: 80% trận cũ để train, 20% trận gần nhất để test.
    # KHÔNG shuffle — xáo trộn ngẫu nhiên cũng là một dạng leakage thời gian.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
    )

    metrics = {
        "so_tran_train": int(len(X_train)),
        "so_tran_test": int(len(X_test)),
        "accuracy": None,
        "log_loss": None,
    }

    if len(X_test) > 20:
        eval_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        eval_model.fit(X_train, y_train)
        preds = eval_model.predict(X_test)
        probs = eval_model.predict_proba(X_test)
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)) * 100, 1)
        metrics["log_loss"] = round(float(log_loss(y_test, probs, labels=[0, 1, 2])), 3)
        logger.info(
            "Đánh giá trên %d trận gần nhất (chưa train): accuracy=%.1f%%, log_loss=%.3f",
            len(X_test), metrics["accuracy"], metrics["log_loss"],
        )
    else:
        logger.warning("Không đủ trận để tách tập test đáng tin cậy, bỏ qua bước đánh giá.")

    # Train lại model CUỐI CÙNG trên toàn bộ dữ liệu (để dùng hết thông tin khi predict thật)
    final_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    final_model.fit(X, y)

    teams = sorted(team_state["elo"].keys())

    return final_model, team_state, teams, metrics


def load_or_train_model(force_retrain: bool = False):
    """
    Nếu đã có cache và còn "mới" (< CACHE_MAX_AGE_SECONDS) thì load lên cho
    nhanh; ngược lại huấn luyện mới rồi lưu cache kèm timestamp.
    """
    if not force_retrain and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)

        age = time.time() - cached.get("trained_at", 0)
        if age < CACHE_MAX_AGE_SECONDS:
            logger.info("Dùng model cache (train cách đây %.1f giờ).", age / 3600)
            return cached["model"], cached["team_state"], cached["teams"], cached["metrics"]
        logger.info("Cache đã cũ (%.1f giờ) — train lại.", age / 3600)

    model, team_state, teams, metrics = train_model()

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "team_state": team_state,
            "teams": teams,
            "metrics": metrics,
            "trained_at": time.time(),
        }, f)

    return model, team_state, teams, metrics


def predict_match(model, team_state: dict, doi_nha: str, doi_khach: str):
    """
    Dự đoán tỉ lệ thắng/hòa/thua cho 1 trận đấu, dùng Elo + form mới nhất
    của mỗi đội (tính đến trận cuối cùng có trong dữ liệu train).

    Trả về dict kết quả, hoặc raise ValueError nếu tên đội không hợp lệ.
    """
    elo = team_state["elo"]
    if doi_nha not in elo:
        raise ValueError(f'Không tìm thấy đội nhà "{doi_nha}" trong dữ liệu.')
    if doi_khach not in elo:
        raise ValueError(f'Không tìm thấy đội khách "{doi_khach}" trong dữ liệu.')

    home_elo = elo[doi_nha]
    away_elo = elo[doi_khach]
    home_form = team_state["home_form"].get(doi_nha, 1.3)
    away_form = team_state["away_form"].get(doi_khach, 1.1)
    home_conceded = team_state["home_conceded"].get(doi_nha, 1.1)
    away_conceded = team_state["away_conceded"].get(doi_khach, 1.3)

    features = pd.DataFrame([{
        "elo_diff": home_elo - away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
        "goal_diff_form": home_form - away_conceded - (away_form - home_conceded),
    }])[FEATURE_COLS]

    probs = model.predict_proba(features)[0]
    # Thứ tự class của model: 0=khách thắng, 1=hoà, 2=nhà thắng (theo _get_result)
    class_order = list(model.classes_)

    def _p(cls):
        return round(float(probs[class_order.index(cls)]) * 100, 1)

    return {
        "doi_nha": doi_nha,
        "doi_khach": doi_khach,
        "thang_nha": _p(2),
        "hoa": _p(1),
        "thang_khach": _p(0),
    }