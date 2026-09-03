"""
player_ratings.py
------------------
Tính rating cầu thủ (thang điểm 0-10) từ số liệu MÙA GIẢI trên FBref
(qua thư viện soccerdata), dùng cho 2 chỗ trên web:
  1. Tab "Cầu thủ": bảng xếp hạng cầu thủ theo giải, lọc theo đội/vị trí.
  2. Kết quả dự đoán trận đấu: liệt kê 3 cầu thủ nổi bật nhất mỗi đội.

CÁCH TÍNH RATING (kết hợp cả 2 hướng theo yêu cầu):
  - Chỉ số tấn công thô (bàn/90p, kiến tạo/90p, xG/90p...) đóng góp cho
    MỌI vị trí, chỉ khác trọng số.
  - Đồng thời rating tổng hợp riêng theo VỊ TRÍ: thủ môn dùng bộ chỉ số
    cản phá/giữ sạch lưới, hậu vệ cộng thêm chỉ số phòng ngự, tiền vệ
    cộng thêm chỉ số chuyền bóng, tiền đạo tăng trọng số các chỉ số tấn công.
  - Mỗi chỉ số được xếp hạng phần trăm (percentile 0-100) so với các cầu
    thủ khác CÙNG giải/mùa, rồi lấy trung bình có trọng số theo vị trí,
    quy về thang 0-10. Cách này tự chuẩn hoá theo dữ liệu thật có, không
    cần biết trước "mức trung bình hợp lý" của từng chỉ số là bao nhiêu.
  - Cầu thủ đá quá ít phút (< MIN_MINUTES) bị loại khỏi rating vì mẫu
    quá nhỏ, dễ gây nhiễu (vd đá 10 phút ghi 1 bàn thì rating tấn công
    sẽ ảo).

LƯU Ý QUAN TRỌNG (giống hệt model.py): FBref cần trình duyệt
(Selenium/Chrome) để cào dữ liệu -> server Render KHÔNG tự tải được.
Phải chạy `python train_players_offline.py` ở máy CÓ Chrome, rồi
commit + push file cache `player_ratings_<giải>.pkl` lên GitHub, đúng
quy trình đang dùng cho model dự đoán trận đấu (xem README.md).

LƯU Ý VỀ TÊN CỘT FBref: soccerdata trả về DataFrame với cột dạng
MultiIndex (vd ("Performance", "Gls")), và tên cột chính xác có thể
thay đổi nhẹ giữa các phiên bản thư viện. Code dưới đây tìm cột theo
TỪ KHOÁ (không hard-code tên cột tuyệt đối) để đỡ vỡ khi lệch phiên
bản — nhưng NÊN chạy debug_player_stats.py sau khi train lần đầu để
kiểm tra xem các chỉ số có nhận diện đúng cột hay không, và chỉnh lại
`METRICS` bên dưới nếu cần.
"""

import os
import time
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import soccerdata as sd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.dirname(__file__)
CACHE_MAX_AGE_SECONDS = 3 * 24 * 60 * 60  # 3 ngày, giống model.py

# Đá tối thiểu ~5 trận đầy đủ mới được tính rating, tránh mẫu quá nhỏ.
MIN_MINUTES = 450

# Các loại bảng số liệu cầu thủ lấy từ FBref (qua soccerdata.FBref).
STAT_TYPES = ("standard", "shooting", "passing", "defense", "possession", "misc", "keeper")

# Các cột dùng để merge nhiều bảng số liệu lại với nhau (1 cầu thủ = 1 dòng).
_ID_COLS = ("league", "season", "team", "player")


def _cache_path(league_key: str) -> str:
    return os.path.join(CACHE_DIR, f"player_ratings_{league_key}.pkl")


# ==================== TIỆN ÍCH ĐỌC CỘT LINH HOẠT ====================

def _flatten_columns(columns) -> List[str]:
    """Biến cột MultiIndex (vd ("Performance", "Gls")) hoặc cột thường
    thành tên chuỗi đơn giản, chữ thường, không dấu cách/%% để dễ tìm
    kiếm theo từ khoá."""
    flat = []
    for col in columns:
        parts = col if isinstance(col, tuple) else (col,)
        cleaned = []
        for p in parts:
            s = str(p).strip()
            if not s or s.lower().startswith("unnamed"):
                continue
            cleaned.append(s)
        name = "_".join(cleaned) if cleaned else "col"
        name = (
            name.lower()
            .replace(" ", "_")
            .replace("%", "pct")
            .replace("-", "_")
            .replace("+", "plus")
            .replace("/", "_")
            .replace(".", "")
        )
        flat.append(name)
    return flat


def _find_col(columns: List[str], keyword_sets: Tuple[Tuple[str, ...], ...],
              exclude_substrings: Tuple[str, ...] = ()) -> Optional[str]:
    """Tìm cột đầu tiên khớp 1 trong các bộ từ khoá (mỗi bộ: TẤT CẢ từ
    khoá phải xuất hiện trong tên cột, dạng token hoặc substring)."""
    for keywords in keyword_sets:
        for col in columns:
            if any(ex in col for ex in exclude_substrings):
                continue
            tokens = col.split("_")
            if all(kw in tokens or kw in col for kw in keywords):
                return col
    return None


# ==================== ĐỊNH NGHĨA CHỈ SỐ & TRỌNG SỐ THEO VỊ TRÍ ====================

# keywords: các bộ từ khoá thử lần lượt để tìm ra đúng cột trong dữ liệu
#           đã gộp (xem _find_col).
# per90:    True nếu đây là số liệu CỘNG DỒN (tổng cả mùa) cần chia cho
#           (số phút / 90) mới ra chỉ số "trung bình mỗi 90 phút".
#           False nếu cột đã là tỉ lệ %% / chỉ số trung bình sẵn.
# inverse:  True nếu chỉ số CÀNG THẤP càng tốt (vd sai lầm, thẻ, bàn thua).
METRICS: Dict[str, Dict[str, Any]] = {
    # ---- Tấn công (áp dụng mọi vị trí, trọng số khác nhau) ----
    "goals":               {"keywords": (("gls",),), "per90": True},
    "assists":             {"keywords": (("ast",),), "per90": True},
    "xg":                  {"keywords": (("npxg",), ("xg",)), "per90": True},
    "shots_on_target_pct": {"keywords": (("sotpct",), ("sot", "pct")), "per90": False},
    "progressive_carries": {"keywords": (("prgc",),), "per90": True},
    "dribbles_pct":        {"keywords": (("succpct",), ("succ", "pct")), "per90": False},
    # ---- Chuyền bóng ----
    "key_passes":          {"keywords": (("kp",),), "per90": True},
    "pass_completion_pct": {"keywords": (("total_cmppct",), ("cmppct",)), "per90": False},
    "progressive_passes":  {"keywords": (("prgp",),), "per90": True},
    # ---- Phòng ngự ----
    "tackles_won":         {"keywords": (("tklw",),), "per90": True},
    "interceptions":       {"keywords": (("int",),), "per90": True},
    "blocks":              {"keywords": (("blocks",),), "per90": True},
    "clearances":          {"keywords": (("clr",),), "per90": True},
    "aerials_won_pct":     {"keywords": (("wonpct",), ("won", "pct")), "per90": False},
    "errors":              {"keywords": (("err",),), "per90": True, "inverse": True},
    # ---- Kỷ luật (trọng số nhỏ, áp dụng nhẹ cho cầu thủ ngoài sân) ----
    "cards_yellow":        {"keywords": (("crdy",),), "per90": True, "inverse": True},
    # ---- Thủ môn ----
    "save_pct":            {"keywords": (("savepct",), ("save", "pct")), "per90": False},
    "goals_against_per90": {"keywords": (("ga90",),), "per90": False, "inverse": True},
    "clean_sheet_pct":     {"keywords": (("cspct",), ("cs", "pct")), "per90": False},
}

# Trọng số theo nhóm vị trí — không cần tổng = 1, code tự chuẩn hoá theo
# các chỉ số THỰC SỰ tìm được cột dữ liệu.
POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "FW": {
        "goals": 3.0, "xg": 1.5, "assists": 1.5, "shots_on_target_pct": 1.0,
        "progressive_carries": 1.0, "key_passes": 1.0, "dribbles_pct": 1.0,
        "cards_yellow": 0.3,
    },
    "MF": {
        "goals": 1.3, "assists": 1.3, "key_passes": 1.3, "pass_completion_pct": 1.3,
        "progressive_passes": 1.3, "tackles_won": 1.0, "interceptions": 1.0,
        "progressive_carries": 0.8, "cards_yellow": 0.3,
    },
    "DF": {
        "tackles_won": 2.0, "interceptions": 2.0, "blocks": 1.0, "clearances": 1.3,
        "pass_completion_pct": 1.3, "aerials_won_pct": 1.0, "errors": 1.0,
        "goals": 0.5, "assists": 0.5, "cards_yellow": 0.4,
    },
    "GK": {
        "save_pct": 3.0, "goals_against_per90": 2.5, "clean_sheet_pct": 2.0,
    },
}

POSITIONS = ("FW", "MF", "DF", "GK")


def _position_bucket(pos_str: str) -> str:
    p = str(pos_str or "").upper()
    if "GK" in p:
        return "GK"
    if "DF" in p:
        return "DF"
    if "FW" in p:
        return "FW"
    if "MF" in p:
        return "MF"
    return "MF"  # không rõ vị trí -> coi như tiền vệ (bộ trọng số cân bằng nhất)


# ==================== TẢI & GỘP DỮ LIỆU TỪ FBREF ====================

def _load_player_frames(league_code: str, season: str) -> Dict[str, pd.DataFrame]:
    """Tải từng loại bảng số liệu cầu thủ từ FBref. Loại nào lỗi thì bỏ
    qua (log warning) chứ không làm hỏng toàn bộ, giống cách _load_raw_games
    xử lý lỗi trong model.py."""
    logger.info(f"Đang tải player stats từ FBref cho {league_code}, mùa {season}...")
    fbref = sd.FBref(leagues=league_code, seasons=[season])

    frames = {}
    for stat_type in STAT_TYPES:
        try:
            df = fbref.read_player_season_stats(stat_type=stat_type)
            df = df.reset_index()
            df.columns = _flatten_columns(df.columns)
            frames[stat_type] = df
            logger.info(f"  '{stat_type}': {len(df)} dòng, {len(df.columns)} cột")
        except Exception as e:
            logger.warning(f"  Không tải được player stats '{stat_type}': {e}")

    if not frames:
        raise ValueError("Không tải được bất kỳ loại số liệu cầu thủ nào từ FBref.")
    return frames


def _merge_player_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gộp các bảng số liệu (standard/shooting/passing/...) thành 1 bảng
    theo (team, player)."""
    base_type = next(iter(frames))
    merged = frames[base_type].copy()

    for stat_type, df in frames.items():
        if stat_type == base_type:
            continue
        merge_cols = [c for c in _ID_COLS if c in merged.columns and c in df.columns]
        if len(merge_cols) < 2:
            logger.warning(f"  Bỏ qua gộp '{stat_type}': không đủ cột định danh chung.")
            continue
        merged = merged.merge(df, on=merge_cols, how="left", suffixes=("", f"__{stat_type}"))

    return merged


def _resolve_minutes_col(cols: List[str]) -> Optional[str]:
    col = _find_col(cols, (("min",),), exclude_substrings=("pct", "age"))
    if col:
        return col
    # dự phòng: cột "90s" (số lần 90 phút đã đá) -> nhân 90 ra số phút
    return _find_col(cols, (("90s",),))


def _resolve_position_col(cols: List[str]) -> Optional[str]:
    return _find_col(cols, (("pos",),), exclude_substrings=("post",))


# ==================== TÍNH RATING ====================

def _compute_ratings(merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = list(merged.columns)

    minutes_col = _resolve_minutes_col(cols)
    if minutes_col is None:
        raise ValueError("Không tìm được cột số phút thi đấu trong dữ liệu FBref.")

    minutes = pd.to_numeric(merged[minutes_col], errors="coerce")
    if "90s" in minutes_col:
        minutes = minutes * 90.0
    merged["_minutes"] = minutes

    pos_col = _resolve_position_col(cols)
    merged["_pos_raw"] = merged[pos_col] if pos_col else ""
    merged["_pos_bucket"] = merged["_pos_raw"].map(_position_bucket)

    resolved_cols: Dict[str, str] = {}
    pct_df = pd.DataFrame(index=merged.index)
    n = len(merged)

    for key, spec in METRICS.items():
        exclude = ("pct90",) if spec.get("per90") else ()
        col = _find_col(cols, spec["keywords"], exclude_substrings=exclude)
        if col is None:
            continue

        series = pd.to_numeric(merged[col], errors="coerce")
        if series.notna().sum() < max(10, int(n * 0.2)):
            continue  # quá ít dữ liệu thực -> bỏ chỉ số này, không đáng tin

        if spec.get("per90"):
            with np.errstate(divide="ignore", invalid="ignore"):
                series = series / (merged["_minutes"] / 90.0)
            series = series.replace([np.inf, -np.inf], np.nan)

        resolved_cols[key] = col
        pct = series.rank(pct=True) * 100.0
        if spec.get("inverse"):
            pct = 100.0 - pct
        pct_df[key] = pct.fillna(50.0)

        # lưu lại giá trị chỉ số thô/90p để hiển thị minh bạch (làm tròn)
        merged[f"_val_{key}"] = series.round(2)

    ratings = pd.Series(np.nan, index=merged.index, dtype=float)
    for bucket, weights in POSITION_WEIGHTS.items():
        mask = merged["_pos_bucket"] == bucket
        if not mask.any():
            continue
        used = {k: w for k, w in weights.items() if k in pct_df.columns}
        total_w = sum(used.values())
        if total_w <= 0:
            continue
        score = sum(pct_df.loc[mask, k] * w for k, w in used.items()) / total_w
        ratings.loc[mask] = (score / 10.0).round(1).clip(0, 10)

    merged["rating"] = ratings

    logger.info(
        "Đã nhận diện %d/%d chỉ số cầu thủ: %s",
        len(resolved_cols), len(METRICS), ", ".join(sorted(resolved_cols)) or "(không có)",
    )
    missing = sorted(set(METRICS) - set(resolved_cols))
    if missing:
        logger.warning(
            "Các chỉ số KHÔNG nhận diện được cột (bị bỏ qua khỏi công thức rating), "
            "hãy kiểm tra lại bằng debug_player_stats.py nếu cần: %s", ", ".join(missing),
        )

    return merged, resolved_cols


def build_player_ratings(league_code: str, season: str) -> Tuple[List[Dict], Dict]:
    """Tải + tính rating cho toàn bộ cầu thủ 1 giải/mùa. Trả về
    (danh sách cầu thủ dạng dict, metadata)."""
    frames = _load_player_frames(league_code, season)
    merged = _merge_player_frames(frames)
    merged, resolved_cols = _compute_ratings(merged)

    merged = merged[merged["_minutes"].fillna(0) >= MIN_MINUTES].copy()

    players = []
    for _, row in merged.iterrows():
        rating = row.get("rating")
        players.append({
            "cau_thu": row.get("player", "?"),
            "doi": row.get("team", "?"),
            "vi_tri": row.get("_pos_bucket", "MF"),
            "vi_tri_goc": row.get("_pos_raw", ""),
            "so_phut": int(row["_minutes"]) if pd.notna(row.get("_minutes")) else 0,
            "ban_thang": _round_or_none(row.get("_val_goals")),
            "kien_tao": _round_or_none(row.get("_val_assists")),
            "rating": float(rating) if pd.notna(rating) else None,
        })

    # Sắp mặc định theo rating giảm dần (None xuống cuối)
    players.sort(key=lambda p: (p["rating"] is None, -(p["rating"] or 0)))

    meta = {
        "season_label": season,
        "built_at": time.time(),
        "resolved_metrics": sorted(resolved_cols),
        "so_cau_thu": len(players),
    }
    return players, meta


def _round_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v))


# ==================== CACHE THEO GIẢI ====================

def load_or_build_player_ratings(
    league_key: str,
    league_code: str,
    season: str,
    force_rebuild: bool = False,
) -> Tuple[List[Dict], Dict]:
    cache_path = _cache_path(league_key)

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            age = time.time() - cached.get("meta", {}).get("built_at", 0)
            if age < CACHE_MAX_AGE_SECONDS:
                logger.info(f"Dùng cache rating cầu thủ của {league_key} (tính cách đây {age/3600:.1f} giờ)")
                return cached["players"], cached["meta"]
            logger.info(f"Cache rating cầu thủ {league_key} cũ ({age/3600:.1f} giờ), tính lại...")
        except Exception as e:
            logger.warning(f"Lỗi đọc cache rating cầu thủ {league_key}: {e}, tính lại...")

    players, meta = build_player_ratings(league_code, season)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"players": players, "meta": meta}, f)
        logger.info(f"Đã lưu cache rating cầu thủ mới cho {league_key}")
    except Exception as e:
        logger.warning(f"Không thể lưu cache rating cầu thủ {league_key}: {e}")

    return players, meta


# ==================== TRUY VẤN / SẮP XẾP (dùng cho API) ====================

PLAYER_SORT_FIELDS = {
    "rating": "Rating",
    "ban_thang": "Bàn thắng",
    "kien_tao": "Kiến tạo",
    "so_phut": "Số phút đã đá",
}


def build_player_leaderboard(players: List[Dict], team: Optional[str] = None,
                              position: Optional[str] = None) -> List[Dict]:
    rows = players
    if team:
        rows = [r for r in rows if r["doi"] == team]
    if position:
        rows = [r for r in rows if r["vi_tri"] == position]
    return [dict(r) for r in rows]  # copy để không sửa nhầm cache gốc


def sort_player_leaderboard(rows: List[Dict], sort_by: str = "rating") -> List[Dict]:
    key = sort_by if sort_by in PLAYER_SORT_FIELDS else "rating"
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get(key) is None, r.get(key) or 0),
        reverse=True,
    )
    for i, r in enumerate(sorted_rows, start=1):
        r["hang"] = i
    return sorted_rows


def get_key_players(players: List[Dict], team: str, limit: int = 3) -> List[Dict]:
    team_players = [dict(p) for p in players if p["doi"] == team and p.get("rating") is not None]
    team_players.sort(key=lambda r: r["rating"], reverse=True)
    return team_players[:limit]
