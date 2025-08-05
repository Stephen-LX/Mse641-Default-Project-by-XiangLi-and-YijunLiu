import os, re, json, random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DataCollatorWithPadding, get_linear_schedule_with_warmup,
    BertConfig
)
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle
from tqdm import tqdm

TRAIN_PATH = os.environ.get("TRAIN_JSONL", "train.jsonl")
VAL_PATH   = os.environ.get("VAL_JSONL",   "val.jsonl")
MODEL_NAME = "bert-base-uncased"

OUT_PT_F1  = os.environ.get("OUT_PT_F1",  "bert_model_bestf1.pt")
OUT_PT_ACC = os.environ.get("OUT_PT_ACC", "bert_model_bestacc.pt")
OUT_LE     = os.environ.get("OUT_LE", "label.pkl")

MAX_LEN          = int(os.environ.get("MAX_LEN", 128))
EPOCHS           = int(os.environ.get("EPOCHS", 20))
BATCH_SIZE_TRAIN = int(os.environ.get("BATCH_SIZE_TRAIN", 8))
BATCH_SIZE_VAL   = int(os.environ.get("BATCH_SIZE_VAL", 16))
LR               = float(os.environ.get("LR", 1e-5))
WEIGHT_DECAY     = float(os.environ.get("WEIGHT_DECAY", 0.02))
WARMUP_RATIO     = float(os.environ.get("WARMUP_RATIO", 0.1))
GRAD_CLIP_NORM   = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
ACCUM_STEPS      = int(os.environ.get("ACCUM_STEPS", 2))
LABEL_SMOOTHING  = float(os.environ.get("LABEL_SMOOTHING", 0.05)) 
USE_AMP          = os.environ.get("USE_AMP", "1") == "1"
WINDOW_STEPS     = int(os.environ.get("WINDOW_STEPS", 10))
PATIENCE         = int(os.environ.get("PATIENCE", 2))
SEED             = int(os.environ.get("SEED", 99))
MIN_TOKENS       = int(os.environ.get("MIN_TOKENS", 1))
DROPOUT_HIDDEN   = float(os.environ.get("DROPOUT_HIDDEN", 0.1))
DROPOUT_ATT      = float(os.environ.get("DROPOUT_ATT", 0.1))

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WS_RE  = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = URL_RE.sub("[URL]", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            post = " ".join(obj.get("postText", []))
            post = clean_text(post)
            if MIN_TOKENS > 1 and len(post.split()) < MIN_TOKENS:
                continue
            label = obj["tags"][0]
            data.append((post, label))
    seen = set(); dedup = []
    for t, y in data:
        key = (t, y)
        if key not in seen:
            seen.add(key); dedup.append((t, y))
    return dedup

class SpoilerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding=False,
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []; total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        y_true.extend(batch["labels"].cpu().numpy())
        y_pred.extend(out.logits.argmax(-1).cpu().numpy())
    return (total/len(loader) if len(loader)>0 else 0.0,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"))

def try_warm_start(model):
    if os.path.exists(OUT_PT_F1):
        state = torch.load(OUT_PT_F1, map_location="cpu")
        model.load_state_dict(state)
        print("Warm start from", OUT_PT_F1)
        return
    if os.path.exists(OUT_PT_ACC):
        state = torch.load(OUT_PT_ACC, map_location="cpu")
        model.load_state_dict(state)
        print("Warm start from", OUT_PT_ACC)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_pairs = load_data(TRAIN_PATH)
    val_pairs   = load_data(VAL_PATH)
    print(f"Train size: {len(train_pairs)} | Val size: {len(val_pairs)}")

    x_train, y_train = zip(*train_pairs)
    x_val,   y_val   = zip(*val_pairs)

    le = None
    if os.path.exists(OUT_LE):
        try:
            with open(OUT_LE, "rb") as f: le = pickle.load(f)
            print("Loaded label encoder:", OUT_LE)
        except Exception: le = None
    if le is None:
        le = LabelEncoder()
        le.fit(list(y_train))
        with open(OUT_LE, "wb") as f: pickle.dump(le, f)
        print("Fitted and saved label encoder:", OUT_LE)

    ytr = le.transform(y_train); yva = le.transform(y_val)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SpoilerDataset(list(x_train), list(ytr), tokenizer, MAX_LEN)
    val_ds   = SpoilerDataset(list(x_val),   list(yva), tokenizer, MAX_LEN)
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_VAL,   shuffle=False, collate_fn=collator)

    cfg = BertConfig.from_pretrained(MODEL_NAME,
                                     hidden_dropout_prob=DROPOUT_HIDDEN,
                                     attention_probs_dropout_prob=DROPOUT_ATT,
                                     num_labels=3)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
    model.to(device)

    try_warm_start(model)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader)) * EPOCHS // max(1, ACCUM_STEPS)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Steps={total_steps}, Warmup={warmup_steps}, LR={LR}, WD={WEIGHT_DECAY}")

    cnt = Counter(ytr); C = len(le.classes_)
    cls_w = torch.tensor([len(ytr)/(cnt[c] if cnt[c]>0 else 1.0) for c in range(C)],
                         device=device, dtype=torch.float32)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_f1, best_acc, bad = -1.0, -1.0, 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        run_loss, win_loss, win_cnt = 0.0, 0.0, 0
        window = max(1, WINDOW_STEPS)
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                    desc=f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=None if LABEL_SMOOTHING>0 else batch["labels"])
                if LABEL_SMOOTHING > 0:
                    loss_true = F.cross_entropy(out.logits, batch["labels"],
                                                weight=cls_w, label_smoothing=LABEL_SMOOTHING)
                else:
                    loss_true = out.loss
            loss = loss_true / max(1, ACCUM_STEPS)
            scaler.scale(loss).backward()
            val_b = float(loss_true.detach().item())
            run_loss += val_b; win_loss += val_b; win_cnt += 1

            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % window == 0:
                pbar.set_postfix(batch_loss=f"{val_b:.4f}",
                                 avg_win_loss=f"{(win_loss/max(1,win_cnt)):.4f}")
                win_loss, win_cnt = 0.0, 0

        tr_avg = run_loss / max(1, len(train_loader))
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_avg:.4f} | val_loss={va_loss:.4f} "
              f"| val_acc={va_acc:.4f} | val_macroF1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, bad = va_f1, 0
            torch.save(model.state_dict(), OUT_PT_F1)
            with open(OUT_LE, "wb") as f: pickle.dump(le, f)
            print(f"[Save] Best F1 -> {best_f1:.4f} | saved: {OUT_PT_F1}, {OUT_LE}")
        else:
            bad += 1

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), OUT_PT_ACC)
            print(f"[Save] Best Acc -> {best_acc:.4f} | saved: {OUT_PT_ACC}")
    
    OUT_PT_LAST = os.environ.get("OUT_PT_LAST", "bert_model_last.pt")
    torch.save(model.state_dict(), OUT_PT_LAST)
    print(f"[Save] Last epoch model -> {OUT_PT_LAST}")
    print(f"Done. Best macro-F1={best_f1:.4f}, Best Acc={best_acc:.4f}")

if __name__ == "__main__":
    main()
