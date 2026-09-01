"""
train_offline.py
----------------
Script chạy offline để huấn luyện model và lưu cache.
Sau khi chạy xong, file model_cache.pkl sẽ được tạo ra.
Bạn có thể copy file này lên Render để dùng ngay.

Cách dùng:
    python train_offline.py [--force]
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
    SEASONS,
    CACHE_MAX_AGE_SECONDS
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train model bóng đá offline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua cache hiện có và train lại từ đầu"
    )
    parser.add_argument(
        "--league",
        type=str,
        default=LEAGUE,
        help=f"Tên giải đấu (mặc định: {LEAGUE})"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=SEASONS,
        help="Danh sách các mùa giải (mặc định: 10 mùa gần nhất)"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=CACHE_PATH,
        help="Đường dẫn lưu cache (mặc định: model_cache.pkl)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU TRAIN MODEL OFFLINE")
    logger.info(f"League: {args.league}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Cache path: {args.cache_path}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Train model (sẽ train mới nếu force hoặc không có cache)
        model, team_state, teams, metrics = train_model(
            league=args.league,
            seasons=args.seasons
        )
        
        # Lưu cache
        cache_data = {
            "model": model,
            "team_state": team_state,
            "teams": teams,
            "metrics": metrics,
            "trained_at": time.time(),
            "version": "2.0",
            "league": args.league,
            "seasons": args.seasons,
        }
        
        with open(args.cache_path, "wb") as f:
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
        logger.info(f"Cache đã lưu tại: {args.cache_path}")
        logger.info("=" * 70)
        
        # In thêm thông tin về models đã đánh giá
        if "models_evaluated" in metrics and metrics["models_evaluated"]:
            logger.info("Chi tiết đánh giá các model:")
            for name, scores in metrics["models_evaluated"].items():
                logger.info(f"  - {name}: acc={scores['accuracy']}%, "
                          f"log_loss={scores['log_loss']}")
        
        # Kiểm tra dung lượng file cache
        cache_size_mb = os.path.getsize(args.cache_path) / (1024 * 1024)
        logger.info(f"Dung lượng cache: {cache_size_mb:.2f} MB")
        
        if cache_size_mb > 100:
            logger.warning("Cache > 100MB, có thể vượt giới hạn free tier của Render (512MB RAM)")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình train: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()