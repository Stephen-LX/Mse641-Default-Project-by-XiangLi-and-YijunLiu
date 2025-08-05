import os, json, glob, pickle
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
import pandas as pd

TEST_PATH  = os.environ.get("TEST_JSONL", "test.jsonl")
MODEL_NAME = "bert-base-uncased"
LABEL_PKL  = os.environ.get("LABEL_PKL", "label.pkl")
OUT_JSONL  = os.environ.get("OUT_JSONL", "T1.jsonl")
OUT_CSV    = os.environ.get("OUT_CSV", "Task1.csv")

MAX_LEN    = int(os.environ.get("MAX_LEN", 128))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
USE_AMP    = os.environ.get("USE_AMP", "1") == "1"
CHECKPOINTS = os.environ.get("CHECKPOINTS", "")
TOP_N       = int(os.environ.get("TOP_N", 0)) 

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

class TestDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer: BertTokenizer, max_len: int):
        self.items = items; self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        post = " ".join(self.items[i].get("postText", []))
        enc = self.tok(post, truncation=True, padding=False,
                       max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

def collect_checkpoints() -> List[str]:
    if CHECKPOINTS.strip():
        cks = [x.strip() for x in CHECKPOINTS.split(",") if x.strip()]
        return [ck for ck in cks if os.path.exists(ck)]
    # 自动搜集：bert_model*.pt
    cks = sorted(glob.glob("bert_model*.pt"))
    if TOP_N > 0:
        cks = cks[:TOP_N]
    if not cks:
        raise FileNotFoundError("No checkpoints found. Set CHECKPOINTS or ensure bert_model*.pt exist.")
    return cks

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_items = load_jsonl(TEST_PATH)
    print("Test size:", len(test_items))

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    ds = TestDataset(test_items, tokenizer, MAX_LEN)
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    with open(LABEL_PKL, "rb") as f:
        le = pickle.load(f)

    ckpts = collect_checkpoints()
    print("Ensembling checkpoints:", ckpts)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device); model.eval()

    all_logits = []
    amp_ctx = torch.cuda.amp.autocast(enabled=USE_AMP)
    with torch.no_grad():
        logits_sum = None
        for ck in ckpts:
            state = torch.load(ck, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.eval()
            batch_logits = []
            pbar = tqdm(dl, desc=f"Predict {os.path.basename(ck)}")
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                with amp_ctx:
                    out = model(**batch)
                    logit = out.logits  # (B,3)
                batch_logits.append(logit.cpu())
            ck_logits = torch.cat(batch_logits, dim=0)
            if logits_sum is None:
                logits_sum = ck_logits
            else:
                logits_sum += ck_logits

    if len(ckpts) > 1:
        logits_sum /= float(len(ckpts))
    preds_idx = logits_sum.argmax(dim=1).numpy().tolist()
    preds_lab = le.inverse_transform(preds_idx).tolist()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for obj, lab in zip(test_items, preds_lab):
            uid = obj.get("uuid") or obj.get("postId")
            f.write(json.dumps({"uuid": uid, "spoilerType": lab}, ensure_ascii=False) + "\n")
    print("Saved", OUT_JSONL)

    rows = [{"id": obj.get("id"), "spoilerType": lab} for obj, lab in zip(test_items, preds_lab)]
    pd.DataFrame(rows, columns=["id","spoilerType"]).to_csv(OUT_CSV, index=False)
    print("Saved", OUT_CSV)

if __name__ == "__main__":
    main()
