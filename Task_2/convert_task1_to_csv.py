import pandas as pd
import json


lines = []
with open('D:/Projects/T2/T2_final.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        lines.append(json.loads(line))
df = pd.DataFrame(lines)
df.rename(columns={'uuid': 'id'}, inplace=True)
df_selected = df[['id', 'spoiler']]
output_filename = 'D:/Projects/T2/T2_final.csv'
df_selected.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Done {output_filename}")
