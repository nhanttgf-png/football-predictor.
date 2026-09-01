"""
debug_team_stats.py
--------------------
Script nhỏ để soi trực tiếp các con số Elo/phong độ/win-rate mà model đang
"nghĩ" về 2 đội bất kỳ, đọc thẳng từ model_cache.pkl vừa train.

Dùng để trả lời câu hỏi: "model dự đoán vậy có hợp lý không, hay bị lỗi?"
mà không cần đoán mò — thấy số liệu thật là biết ngay.

Cách dùng:
    python debug_team_stats.py "Arsenal" "Manchester City"
"""

import sys
import pickle
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "model_cache.pkl")


def main():
    if len(sys.argv) != 3:
        print('Cách dùng: python debug_team_stats.py "Đội A" "Đội B"')
        sys.exit(1)

    team_a, team_b = sys.argv[1], sys.argv[2]

    if not os.path.exists(CACHE_PATH):
        print(f"Không tìm thấy {CACHE_PATH}. Hãy chạy train_offline.py trước.")
        sys.exit(1)

    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)

    team_state = cached["team_state"]
    metrics = cached.get("metrics", {})
    teams = cached.get("teams", [])

    print("=" * 70)
    print(f"Model type đang dùng : {metrics.get('model_type')}")
    print(f"Accuracy (tập test)  : {metrics.get('accuracy')}%")
    print(f"Log loss (tập test)  : {metrics.get('log_loss')}")
    print(f"Số trận train/test   : {metrics.get('so_tran_train')} / {metrics.get('so_tran_test')}")
    print("=" * 70)

    for team in (team_a, team_b):
        if team not in teams:
            print(f'\n⚠️  Không tìm thấy đội "{team}" trong dữ liệu.')
            print(f"Các đội có sẵn (gần giống): "
                  f"{[t for t in teams if team.lower()[:4] in t.lower()]}")
            continue

        elo = team_state["elo"].get(team)
        home_gp = team_state.get("home_games_played", {}).get(team, 0)
        away_gp = team_state.get("away_games_played", {}).get(team, 0)
        home_pts = team_state.get("home_points", {}).get(team, 0)
        away_pts = team_state.get("away_points", {}).get(team, 0)
        home_ppg = home_pts / home_gp if home_gp else None
        away_ppg = away_pts / away_gp if away_gp else None
        home_unbeaten = team_state.get("home_unbeaten", {}).get(team, 0)
        away_unbeaten = team_state.get("away_unbeaten", {}).get(team, 0)
        last_played = team_state.get("last_played", {}).get(team)

        print(f"\n📊 {team}")
        print(f"   Elo hiện tại          : {elo:.0f}" if elo else "   Elo hiện tại          : (chưa có)")
        print(f"   Điểm/trận (sân nhà)   : {home_ppg:.2f} ({home_gp} trận)" if home_ppg is not None else "   Điểm/trận (sân nhà)   : (chưa đá trận nào ghi nhận)")
        print(f"   Điểm/trận (sân khách) : {away_ppg:.2f} ({away_gp} trận)" if away_ppg is not None else "   Điểm/trận (sân khách) : (chưa đá trận nào ghi nhận)")
        print(f"   Chuỗi bất bại sân nhà : {home_unbeaten} trận")
        print(f"   Chuỗi bất bại sân khách: {away_unbeaten} trận")
        print(f"   Đá gần nhất           : {last_played}")

    print("\n" + "=" * 70)
    elo_a = team_state["elo"].get(team_a)
    elo_b = team_state["elo"].get(team_b)
    if elo_a is not None and elo_b is not None:
        print(f"Chênh lệch Elo ({team_a} - {team_b}): {elo_a - elo_b:+.0f}")
        print("(Chênh lệch dương nghĩa là model đang đánh giá "
              f"{team_a} mạnh hơn {team_b} ở thời điểm hiện tại, "
              "không tính đến các mùa cũ trước khi có sự thay đổi phong độ.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
