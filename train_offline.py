"""
train_offline.py
----------------
Script chạy offline để huấn luyện model và lưu cache.
Sau khi chạy xong, file model_cache*.pkl sẽ được tạo ra.
Bạn có thể copy file này lên Render để dùng ngay.

Cách dùng (KHUYẾN NGHỊ — dùng --league-key, không tự phải nhớ đường dẫn
cache, tool tự chọn ĐÚNG file cho từng giải, không sợ ghi đè nhầm):
    python train_offline.py --league-key premier-league
    python train_offline.py --league-key la-liga
    python train_offline.py --league-key serie-a
    python train_offline.py --league-key bundesliga
    python train_offline.py --league-key ligue-1

(Cách cũ dùng --league "ENG-Premier League" vẫn còn hỗ trợ để tương thích
ngược, và giờ cũng tự suy ra đúng cache-path tương ứng nếu nhận diện được
tên giải — nhưng --league-key vẫn AN TOÀN HƠN vì không phụ thuộc việc gõ
đúng tên giải theo FBref.)
"""

import os
import sys
import time
import argparse
import logging
import pickle

# Ép stdout/stderr dùng UTF-8, tránh lỗi UnicodeEncodeError khi in tiếng Việt
# có dấu trên terminal Windows (mặc định dùng cp1252, không hỗ trợ Unicode).
# logging module cũng ghi qua stderr nên cần fix ở đây, không chỉ với print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import các hàm từ model.py
from model import (
    train_model,
    load_or_train_model,
    CACHE_PATH,
    LEAGUE,
    LEAGUES,
    DEFAULT_LEAGUE_KEY,
    SEASONS,
    CACHE_MAX_AGE_SECONDS,
    _cache_path,
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def _resolve_league_key_by_code(code: str):
    """Tìm league_key (vd 'la-liga') tương ứng với 1 mã FBref (vd
    'ESP-La Liga'), dùng cho trường hợp người dùng vẫn gõ --league kiểu
    cũ thay vì --league-key. Trả về None nếu không khớp giải nào."""
    for key, cfg in LEAGUES.items():
        if cfg["code"] == code:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(description="Train model bóng đá offline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua cache hiện có và train lại từ đầu"
    )
    parser.add_argument(
        "--league-key",
        type=str,
        choices=list(LEAGUES.keys()),
        default=None,
        help=(
            "KHUYẾN NGHỊ DÙNG THAM SỐ NÀY: khoá giải đấu như web dùng "
            f"({', '.join(LEAGUES.keys())}). Tool sẽ tự suy ra đúng tên "
            "giải (FBref) và đúng file cache tương ứng, tránh ghi đè nhầm "
            "cache của giải khác."
        ),
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help=(
            "(Cách cũ) Tên giải đấu theo FBref, vd \"ENG-Premier League\". "
            "Chỉ dùng khi không dùng --league-key. Nếu bỏ trống cả hai, "
            f"mặc định là {LEAGUE}."
        )
    )
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=None,
        help="Danh sách các mùa giải (mặc định: theo cấu hình của giải đó)"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help=(
            "Đường dẫn lưu cache. Nếu bỏ trống, tool TỰ CHỌN đúng file "
            "theo giải đấu (vd model_cache_la-liga.pkl) -- chỉ cần tự "
            "truyền tay tham số này nếu bạn thật sự muốn ghi ra vị trí "
            "khác với mặc định."
        )
    )

    args = parser.parse_args()

    # ============ SUY RA league_key / league_code / cache_path ĐÚNG ========
    if args.league_key:
        league_key = args.league_key
        league_code = LEAGUES[league_key]["code"]
    elif args.league:
        league_code = args.league
        league_key = _resolve_league_key_by_code(league_code)
        if league_key is None:
            logger.warning(
                "Không nhận diện được \"%s\" khớp với giải nào trong danh "
                "sách web đang hỗ trợ (%s). Sẽ KHÔNG tự suy ra được cache-path "
                "đúng -- nếu bạn không tự truyền --cache-path, cache sẽ được "
                "lưu vào %s (mặc định của Ngoại hạng Anh), có thể ghi đè "
                "NHẦM cache của giải khác!",
                league_code, ", ".join(LEAGUES.keys()), CACHE_PATH,
            )
    else:
        league_key = DEFAULT_LEAGUE_KEY
        league_code = LEAGUE

    seasons = args.seasons or LEAGUES.get(league_key, {}).get("seasons", SEASONS)

    if args.cache_path:
        cache_path = args.cache_path
    elif league_key is not None:
        cache_path = _cache_path(league_key)
    else:
        cache_path = CACHE_PATH

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU TRAIN MODEL OFFLINE")
    logger.info(f"Giải đấu (key): {league_key or '(không xác định)'}")
    logger.info(f"Giải đấu (FBref code): {league_code}")
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Cache sẽ lưu tại: {cache_path}")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # Train model (sẽ train mới nếu force hoặc không có cache)
        model, team_state, teams, metrics = train_model(
            league=league_code,
            seasons=seasons
        )

        # Lưu cache
        cache_data = {
            "model": model,
            "team_state": team_state,
            "teams": teams,
            "metrics": metrics,
            "trained_at": time.time(),
            "version": "2.0",
            "league": league_code,
            "league_key": league_key,
            "seasons": seasons,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info("HOÀN THÀNH TRAIN MODEL OFFLINE")
        logger.info(f"Thời gian train: {elapsed/60:.2f} phút")
        logger.info(f"Số đội: {len(teams)}")
        logger.info(f"Model type: {metrics.get('model_type', 'unknown')}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A')}%")
        logger.info(f"Log loss: {metrics.get('log_loss', 'N/A')}")
        logger.info(f"Số trận train: {metrics.get('so_tran_train', 'N/A')}")
        logger.info(f"Số trận test: {metrics.get('so_tran_test', 'N/A')}")
        logger.info(f"Cache đã lưu tại: {cache_path}")
        logger.info("=" * 70)

        # In thêm thông tin về models đã đánh giá
        if "models_evaluated" in metrics and metrics["models_evaluated"]:
            logger.info("Chi tiết đánh giá các model:")
            for name, scores in metrics["models_evaluated"].items():
                logger.info(f"  - {name}: acc={scores['accuracy']}%, "
                          f"log_loss={scores['log_loss']}")

        # Kiểm tra dung lượng file cache
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        logger.info(f"Dung lượng cache: {cache_size_mb:.2f} MB")

        if cache_size_mb > 100:
            logger.warning("Cache > 100MB, có thể vượt giới hạn free tier của Render (512MB RAM)")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình train: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()