import json, sys
sys.stdout.reconfigure(encoding='utf-8')
d = json.load(open('FinalData/penalty_data/all_penalty_history_enriched.json', encoding='utf-8'))
for p in d['856250']['penalties']:
    if 'enriched' in p:
        print(json.dumps(p['enriched'], indent=2, ensure_ascii=False))
        break
