import json
import pandas as pd
from pathlib import Path

# 경로 설정
repo_root = Path(".")
player_stats_file = repo_root / "data/player_stats_final.csv"
needed_spids_file = repo_root / "data/winrate/needed_spids.json"

# 1. player_stats_final.csv에서 이미 있는 spId 로드
df_players = pd.read_csv(player_stats_file)
existing_spids = set(df_players['spid'].tolist())
print(f"✓ 이미 있는 spId: {len(existing_spids)}개")

# 2. needed_spids.json 로드
with open(needed_spids_file, 'r', encoding='utf-8') as f:
    all_needed = set(json.load(f))
print(f"✓ 전체 필요한 spId: {len(all_needed)}개")

# 3. 중복 제거 (이미 있는 것 제외)
truly_needed = all_needed - existing_spids
print(f"✓ 정말 필요한 spId (제외 후): {len(truly_needed)}개")

# 4. 정렬해서 저장
truly_needed_list = sorted(list(truly_needed))

with open(needed_spids_file, 'w', encoding='utf-8') as f:
    json.dump(truly_needed_list, f, ensure_ascii=False, indent=2)

print(f"\n✓ needed_spids.json 업데이트 완료")
print(f"  - 팀원이 수집해야 할 spId: {len(truly_needed_list)}개")
print(f"  - 처음 10개: {truly_needed_list[:10]}")
print(f"  - 마지막 10개: {truly_needed_list[-10:]}")
