"""
train_players_offline.py
-------------------------
Script chạy offline để tính rating cầu thủ (từ số liệu FBref) và lưu
cache `player_ratings_<giải>.pkl`. Y hệt vai trò của train_offline.py
với model dự đoán trận đấu: CHẠY Ở MÁY LOCAL CÓ CHROME, vì FBref cần
trình duyệt để cào dữ liệu, Render (free tier) không có Chrome.

Cách dùng:
    # 1 giải (mặc định Ngoại hạng Anh, mùa liền trước mùa hiện tại):
    python train_players_offline.py

    # chọn giải + mùa cụ thể:
    python train_players_offline.py --league "ESP-La Liga" --season 2425

    # tính cho TẤT CẢ giải đang hỗ trợ trong 1 lần chạy:
    python train_players_offline.py --all

Sau khi chạy xong, commit + push các file player_ratings_<giải>.pkl mới
lên GitHub để Render dùng được (giống model_cache_<giải>.pkl).
"""

import sys
import time
import argparse
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import LEAGUES, DEFAULT_LEAGUE_KEY
from player_ratings import load_or_build_player_ratings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _default_season(league_key: str) -> str:
    seasons = LEAGUES[league_key]["seasons"]
    return seasons[-2] if len(seasons) >= 2 else seasons[-1]


def _run_one(league_key: str, season: str):
    cfg = LEAGUES[league_key]
    logger.info("=" * 70)
    logger.info(f"TÍNH RATING CẦU THỦ: {cfg['name']} ({cfg['code']}), mùa {season}")
    logger.info("=" * 70)

    start = time.time()
    players, meta = load_or_build_player_ratings(
        league_key=league_key,
        league_code=cfg["code"],
        season=season,
        force_rebuild=True,
    )
    elapsed = time.time() - start

    logger.info(f"Xong trong {elapsed/60:.2f} phút — {len(players)} cầu thủ đủ điều kiện.")
    logger.info(f"Các chỉ số nhận diện được: {', '.join(meta.get('resolved_metrics', []))}")

    top5 = [p for p in players if p.get("rating") is not None][:5]
    if top5:
        logger.info("Top 5 rating cao nhất:")
        for p in top5:
            logger.info(f"  {p['rating']:.1f} - {p['cau_thu']} ({p['doi']}, {p['vi_tri']})")


def main():
    parser = argparse.ArgumentParser(description="Tính rating cầu thủ offline")
    parser.add_argument("--league", type=str, default=None,
                         help='Mã giải theo FBref, vd "ENG-Premier League" (mặc định: tất cả nếu dùng --all, '
                              f'ngược lại {LEAGUES[DEFAULT_LEAGUE_KEY]["code"]})')
    parser.add_argument("--season", type=str, default=None,
                         help="Mùa giải (vd 2425). Mặc định: mùa liền trước mùa hiện tại của giải đó.")
    parser.add_argument("--all", action="store_true",
                         help="Tính cho tất cả giải đấu đang hỗ trợ (bỏ qua --league)")
    args = parser.parse_args()

    if args.all:
        for league_key in LEAGUES:
            season = args.season or _default_season(league_key)
            try:
                _run_one(league_key, season)
            except Exception as e:
                logger.error(f"Lỗi khi tính rating cho {league_key}: {e}", exc_info=True)
        return

    # Tìm league_key khớp với --league (theo code) hoặc dùng mặc định
    league_key = DEFAULT_LEAGUE_KEY
    if args.league:
        match = next((k for k, v in LEAGUES.items() if v["code"] == args.league), None)
        if match is None:
            logger.error(
                f'Không nhận ra giải "{args.league}". Các mã hợp lệ: '
                f'{", ".join(v["code"] for v in LEAGUES.values())}'
            )
            sys.exit(1)
        league_key = match

    season = args.season or _default_season(league_key)

    try:
        _run_one(league_key, season)
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tính rating: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
