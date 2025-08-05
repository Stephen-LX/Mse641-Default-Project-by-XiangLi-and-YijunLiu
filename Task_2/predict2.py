# predict.py
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm 

MODEL_PATH = "D:/my_model/lar_bart_pas_and_pha_modelv2"
INPUT_FILE = "D:/Projects/T2/data_combined_for_testv3.jsonl" 
OUTPUT_FILE = "D:/Projects/T2/T2_final.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval() 
    print("Model sucess")
except OSError:
    print(f" '{MODEL_PATH}' cannot find")
    exit()

def generate_prediction(post_text, target_paragraphs, tag):
    
    input_text = f"指令: {tag}\n正文: {post_text}\n段落: {' '.join(target_paragraphs)}"
    
    inputs = tokenizer(
        input_text, 
        max_length=512, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            max_length=256,
            early_stopping=True
        )
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

print(f"start {INPUT_FILE}")

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for index, line in enumerate(tqdm(lines, desc="predicting")):
            if not line.strip():
                continue
            
            sample = json.loads(line)
            post = " ".join(sample.get("postText", []))
            paragraphs = sample.get("targetParagraphs", [])
            tag = sample.get("tag", "unknown") 
            uuid = sample.get("uuid", "no-uuid") 
            predicted_spoiler = generate_prediction(post, paragraphs, tag)
            result_to_write = {
                "id": index,  
                "spoiler": predicted_spoiler 
            }
            out_f.write(json.dumps(result_to_write, ensure_ascii=False) + '\n')

except FileNotFoundError:
    print(f" '{INPUT_FILE}' Not exit。")
except Exception as e:
    print("Error {e}")