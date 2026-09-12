import json
import pandas as pd
from pathlib import Path

# 경로 설정
repo_root = Path(".")
player_stats_file = repo_root / "data/player_stats_final.csv"
needed_spids_file = repo_root / "data/winrate/needed_spids.json"
matches_file = repo_root / "data/winrate/matches.jsonl"

SUB_POSITION_CODE = 28

# 1. player_stats_final.csv에서 이미 있는 spId 로드
df_players = pd.read_csv(player_stats_file)
existing_spids = set(df_players['spid'].tolist())
print(f"✓ player_stats_final.csv의 spId: {len(existing_spids)}개")

# 2. matches.jsonl에서 SUB 제외하고 추출 (제대로)
extracted_spids = set()
with open(matches_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = json.loads(line)
        for info in match.get('matchInfo', []):
            for p in info.get('player', []):
                if p.get('spPosition') != SUB_POSITION_CODE:
                    extracted_spids.add(p.get('spId'))

print(f"✓ matches.jsonl에서 추출한 spId (SUB 제외): {len(extracted_spids)}개")

# 3. 교집합 확인
overlap = existing_spids & extracted_spids
print(f"\n교집합 (둘 다 있는 것): {len(overlap)}개")
print(f"매칭률: {len(overlap) / len(extracted_spids) * 100:.1f}%")

# 4. 실제 필요한 것 (matches에만 있는 것)
truly_needed = extracted_spids - existing_spids
print(f"\n정말 필요한 spId (matches에만 있는 것): {len(truly_needed)}개")

# 5. needed_spids.json 업데이트 (SUB 제외)
truly_needed_list = sorted(list(truly_needed))
with open(needed_spids_file, 'w', encoding='utf-8') as f:
    json.dump(truly_needed_list, f, ensure_ascii=False, indent=2)

print(f"\n✓ needed_spids.json 업데이트 완료 (SUB 제외)")
print(f"  - 팀원이 수집해야 할 spId: {len(truly_needed_list)}개")
