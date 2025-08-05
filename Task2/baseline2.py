import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, get_scheduler
from torch.optim import AdamW 

MODEL_CHECKPOINT = "D:/my_model/lar_bart_pas_and_pha_modelv2"
TRAIN_FILE = 'train.jsonl'
VAL_FILE = 'val.jsonl'
TARGET_TAG = "passage"
BASE_MODEL = "facebook/bart-large"               
SAVE_PATH= "D:/my_model/lar_bart_pas_and_pha_modelv3"
# 训练超参数
BATCH_SIZE = 8          
NUM_EPOCHS = 3      
LEARNING_RATE = 2e-5   
MAX_INPUT_LENGTH = 512  
MAX_TARGET_LENGTH = 256 

ALLOWED_TASK_TAGS = ["passage", "phrase"]
def filter_data_in_memory(file_path, allowed_tags):
    print(f"--- start: {file_path} ---")
    print(f"Goal tag: {allowed_tags}")
    
    filtered_samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                current_tags = data.get("tags")
                if isinstance(current_tags, list) and current_tags:
                    if "multi" in current_tags:
                        continue
                    if set(current_tags) & set(allowed_tags):
                        filtered_samples.append(data)
                        
    except FileNotFoundError:
        print(f"!!! 数据文件 {file_path} 未找到!")
        return None
    except json.JSONDecodeError:
        print(f"!!! 文件 {file_path} 中有无法解析的JSON行，已跳过。")


    print(f"fINISH 从 {file_path} 中找到 {len(filtered_samples)} 条样本。")
    return filtered_samples

class PassageDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.tokenizer = tokenizer
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]

        tag_list = item.get('tags', ['unknown'])
        tag = tag_list[0] if tag_list else 'unknown'

        post_text = " ".join(item.get('postText', []))
        all_paragraphs = item.get('targetParagraphs', [])
        
        spoiler_data = item.get('spoiler', '')

        if isinstance(spoiler_data, list):
            spoiler_text = spoiler_data[0] if spoiler_data else ""
        else:
            spoiler_text = str(spoiler_data)
      
        golden_paragraph = ""
        if all_paragraphs:
            golden_paragraph = all_paragraphs[0]
            if spoiler_text:
                for p in all_paragraphs:
                    if spoiler_text.strip() in p.strip():
                        golden_paragraph = p
                        break
        
        input_text = f"指令: {tag}\n正文: {post_text}\n段落: {golden_paragraph}"
        target_text = spoiler_text

        model_inputs = self.tokenizer(input_text, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True,return_tensors="pt")
        labels = self.tokenizer(text_target=target_text, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True,return_tensors="pt").input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": model_inputs.input_ids.squeeze(), "attention_mask": model_inputs.attention_mask.squeeze(), "labels": labels.squeeze()}

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ALLOWED_TASK_TAGS = ["passage", "phrase"]
    train_data_list = filter_data_in_memory(TRAIN_FILE, ALLOWED_TASK_TAGS)
    val_data_list = filter_data_in_memory(VAL_FILE, ALLOWED_TASK_TAGS)
    
    if not train_data_list: return

    if os.path.exists(MODEL_CHECKPOINT):
        print(f"用已保存版")
        model_load_path = MODEL_CHECKPOINT
    else:
        print(f"--- 未发现已保存的模型 ---")
        model_load_path = BASE_MODEL
    
    tokenizer = BartTokenizer.from_pretrained(model_load_path)
    model = BartForConditionalGeneration.from_pretrained(model_load_path)
    model.to(device)

    train_dataset = PassageDataset(train_data_list, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = PassageDataset(val_data_list, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    best_em_score = 0.0 

    print(f"\n--- 开始训练，共 {NUM_EPOCHS} 轮 ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\n>>> Epoch {epoch + 1}/{NUM_EPOCHS} <<<")
        model.train()
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        model.eval()
        total_exact_match = 0
        total_loss = 0
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                generated_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=MAX_TARGET_LENGTH)
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            for pred, ref in zip(preds, refs):
                if pred.strip() == ref.strip():
                    total_exact_match += 1
        
        avg_val_loss = total_loss / len(val_dataloader)
        accuracy = total_exact_match / len(val_dataset)
        print(f" (Validation Loss): {avg_val_loss:.4f}")
        print(f"(Validation Exact Match): {accuracy:.4f}")

        if accuracy > best_em_score:
            print(f"从 {best_em_score:.4f} 提升至 {accuracy:.4f}。正在保存模型... !!!")
            best_em_score = accuracy # 更新最高分
            # 只有在分数更高时才保存
            if not os.path.exists(MODEL_CHECKPOINT):
                os.makedirs(MODEL_CHECKPOINT)
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(MODEL_CHECKPOINT)
            print("模型已成功保存！")
        else:
            print(f"本轮分数 {accuracy:.4f} 未超过历史最高分 {best_em_score:.4f}，不保存模型。")
        # ===============================================================

if __name__ == "__main__":
    train_and_evaluate()