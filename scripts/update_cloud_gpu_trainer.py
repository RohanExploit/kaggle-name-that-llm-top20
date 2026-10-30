import json

with open("notebooks/02_kaggle_colab_gpu_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "MODEL_NAME = 'microsoft/deberta-v3-small'" in src:
            cell["source"] = [
                "# 5. Transformer Model Training Setup\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import numpy as np\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup\n",
                "from sklearn.metrics import balanced_accuracy_score\n",
                "\n",
                "MODEL_NAME = 'microsoft/deberta-v3-small'\n",
                "MAX_LEN = 256\n",
                "BATCH_SIZE = 8\n",
                "LR = 2.5e-5\n",
                "EPOCHS = 6\n",
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                "\n",
                "class NTLDataset(Dataset):\n",
                "    def __init__(self, texts, labels, tokenizer, max_len=256):\n",
                "        self.texts = list(texts)\n",
                "        self.labels = list(labels) if labels is not None else None\n",
                "        self.tokenizer = tokenizer\n",
                "        self.max_len = max_len\n",
                "    def __len__(self):\n",
                "        return len(self.texts)\n",
                "    def __getitem__(self, idx):\n",
                "        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')\n",
                "        item = {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0)}\n",
                "        if self.labels is not None:\n",
                "            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)\n",
                "        return item\n",
                "\n",
                "print(f'Ready to train {MODEL_NAME} on {DEVICE}!')\n"
            ]
        elif "tr_loss += loss.item()" in src:
            # Training loop cell
            cell["source"] = [
                "# 6. Class-Weighted Balanced Training with Checkpointing\n",
                "train_sub = df[df['fold'] != 0].reset_index(drop=True)\n",
                "val_sub = df[df['fold'] == 0].reset_index(drop=True)\n",
                "\n",
                "# Compute inverse frequency class weights to prevent majority collapse\n",
                "class_counts = train_sub['label_id'].value_counts().sort_index().values\n",
                "weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(DEVICE)\n",
                "weights = (weights / weights.sum()) * 18.0\n",
                "criterion = nn.CrossEntropyLoss(weight=weights)\n",
                "\n",
                "train_loader = DataLoader(NTLDataset(train_sub['text'], train_sub['label_id'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)\n",
                "val_loader = DataLoader(NTLDataset(val_sub['text'], val_sub['label_id'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)\n",
                "\n",
                "model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=18).to(DEVICE)\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6, weight_decay=0.01)\n",
                "scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*0.1), num_training_steps=len(train_loader)*EPOCHS)\n",
                "\n",
                "best_acc = 0.0\n",
                "print(f'Starting class-weighted training for {EPOCHS} epochs on {DEVICE}...')\n",
                "for epoch in range(EPOCHS):\n",
                "    model.train()\n",
                "    tr_loss = 0.0\n",
                "    for batch in train_loader:\n",
                "        optimizer.zero_grad()\n",
                "        ids, mask, y = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)\n",
                "        out = model(ids, attention_mask=mask)\n",
                "        loss = criterion(out.logits.float(), y)\n",
                "        if torch.isnan(loss):\n",
                "            continue\n",
                "        loss.backward()\n",
                "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                "        optimizer.step()\n",
                "        scheduler.step()\n",
                "        tr_loss += loss.item()\n",
                "    \n",
                "    model.eval()\n",
                "    preds, targets = [], []\n",
                "    with torch.no_grad():\n",
                "        for batch in val_loader:\n",
                "            ids, mask, y = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)\n",
                "            out = model(ids, attention_mask=mask)\n",
                "            preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())\n",
                "            targets.extend(y.cpu().numpy())\n",
                "    bal_acc = balanced_accuracy_score(targets, preds)\n",
                "    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {tr_loss/len(train_loader):.4f} | Val Balanced Acc: {bal_acc:.4f}')\n",
                "    if bal_acc > best_acc:\n",
                "        best_acc = bal_acc\n",
                "        model.save_pretrained('./saved_model')\n",
                "        tokenizer.save_pretrained('./saved_model')\n",
                "        print(f'New Best Model saved with Balanced Acc: {best_acc:.4f}')\n",
                "\n",
                "print(f'Training complete! Peak Balanced Acc: {best_acc:.4f}')\n"
            ]

with open("notebooks/02_kaggle_colab_gpu_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 02_kaggle_colab_gpu_train.ipynb with class-weighting and best checkpointing!")
