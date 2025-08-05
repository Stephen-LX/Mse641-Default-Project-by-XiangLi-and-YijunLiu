import json

PREDICTION_FILE = 'D:/Projects/T2/T1 (10).jsonl' 
ORIGINAL_FILE = 'D:/Projects/T2/test.jsonl'  
OUTPUT_FILE = 'D:/Projects/T2/data_combined_for_testv3.jsonl'

spoiler_type_list = []
try:
    with open(PREDICTION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if 'spoilerType' in data:
                spoiler_type_list.append(data['spoilerType'])
            else:
                spoiler_type_list.append(None) 
except FileNotFoundError:
    print(f"NotFIND {PREDICTION_FILE}")
    exit()

if not spoiler_type_list:
    print(f"{PREDICTION_FILE} No spoiler type")
    exit()

print(f"successful {len(spoiler_type_list)}  spoilerType。")


combined_data = []
processed_count = 0

try:
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if not line.strip(): continue
            if index >= len(spoiler_type_list):
                print(f" {ORIGINAL_FILE} over {PREDICTION_FILE}。")
                print(f"stop{index} ")
                break

            original_data = json.loads(line)
            new_data_point = original_data.copy()
            new_data_point['tag'] = spoiler_type_list[index]
            new_data_point['id'] = index

            combined_data.append(new_data_point)
            processed_count += 1

except FileNotFoundError:
    print(f"no {ORIGINAL_FILE}。")
    exit()
except json.JSONDecodeError as e:
    exit()

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in combined_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Done in {OUTPUT_FILE}")