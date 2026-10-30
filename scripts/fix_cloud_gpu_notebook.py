import json

with open("notebooks/02_kaggle_colab_gpu_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 6 is index 5
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "scaler.scale(loss).backward()" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "scaler = torch.cuda.amp.GradScaler" in line:
                new_source.append("    # Scaler omitted for PyTorch 2.11+ compatibility\n")
            elif "scaler.scale(loss).backward()" in line:
                new_source.append("        loss.backward()\n")
                new_source.append("        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n")
            elif "scaler.step(optimizer)" in line:
                new_source.append("        optimizer.step()\n")
            elif "scaler.update()" in line:
                continue
            else:
                new_source.append(line)
        cell["source"] = new_source

with open("notebooks/02_kaggle_colab_gpu_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed notebooks/02_kaggle_colab_gpu_train.ipynb cleanly!")
