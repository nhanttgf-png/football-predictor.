"""
debug_player_stats.py
----------------------
Soi nhanh cache rating cầu thủ vừa tính từ train_players_offline.py:
top N cầu thủ theo rating, và danh sách chỉ số đã/chưa nhận diện được
cột dữ liệu (hữu ích để chỉnh lại METRICS trong player_ratings.py nếu
tên cột FBref thực tế khác với từ khoá đang tìm).

Cách dùng:
    python debug_player_stats.py                      # Ngoại hạng Anh, top 20
    python debug_player_stats.py --league la-liga --top 10
    python debug_player_stats.py --team "Arsenal"      # chỉ 1 đội
"""

import argparse
import pickle
import os
import sys

from model import LEAGUES, DEFAULT_LEAGUE_KEY
from player_ratings import _cache_path, PLAYER_SORT_FIELDS  # noqa: F401 (dùng nội bộ để debug)


def main():
    parser = argparse.ArgumentParser(description="Soi cache rating cầu thủ")
    parser.add_argument("--league", type=str, default=DEFAULT_LEAGUE_KEY, help="Key giải đấu, vd la-liga")
    parser.add_argument("--team", type=str, default=None, help="Lọc theo 1 đội")
    parser.add_argument("--top", type=int, default=20, help="Số dòng hiển thị")
    args = parser.parse_args()

    if args.league not in LEAGUES:
        print(f'Giải "{args.league}" không hợp lệ. Các key hợp lệ: {", ".join(LEAGUES)}')
        sys.exit(1)

    cache_path = _cache_path(args.league)
    if not os.path.exists(cache_path):
        print(f"Không tìm thấy {cache_path}. Hãy chạy train_players_offline.py trước.")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)

    players = cached["players"]
    meta = cached["meta"]

    print("=" * 70)
    print(f"Giải: {LEAGUES[args.league]['name']} · Mùa: {meta.get('season_label')}")
    print(f"Tổng số cầu thủ đủ điều kiện (đá đủ số phút tối thiểu): {meta.get('so_cau_thu')}")
    print(f"Chỉ số ĐÃ nhận diện được cột dữ liệu: {', '.join(meta.get('resolved_metrics', [])) or '(không có)'}")
    print("=" * 70)

    rows = players
    if args.team:
        rows = [p for p in rows if args.team.lower() in p["doi"].lower()]
        if not rows:
            print(f'Không tìm thấy đội nào khớp "{args.team}".')
            sys.exit(0)

    print(f"\nTop {args.top} theo rating"
          + (f' (đội: {args.team})' if args.team else "") + ":\n")
    for p in rows[: args.top]:
        rating = f"{p['rating']:.1f}" if p.get("rating") is not None else " - "
        print(
            f"  {rating:>4}  {p['cau_thu']:<28} {p['doi']:<22} "
            f"{p['vi_tri']:<3} {p['so_phut']:>5} phút  "
            f"{p.get('ban_thang') or 0} bàn  {p.get('kien_tao') or 0} kiến tạo"
        )


if __name__ == "__main__":
    main()
