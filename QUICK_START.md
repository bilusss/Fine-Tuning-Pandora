# 🚀 Szybki Start - Fine-Tuning Mistral-7B

## 1️⃣ Podstawowe uruchomienie (ZALECANE)

```bash
# Aktywuj środowisko wirtualne (jeśli potrzeba)
source .venv/bin/activate

# Uruchom trening
.venv/bin/python run.py
```

## 2️⃣ Z logowaniem do pliku

```bash
./run_safe.sh
```

## 3️⃣ Ultra-safe mode (jeśli problemy z pamięcią)

```bash
.venv/bin/python run_ultra_safe.py
```

## 📊 Monitorowanie podczas treningu

### Terminal 1: Trening
```bash
.venv/bin/python run.py
```

### Terminal 2: Monitor GPU
```bash
watch -n 1 nvidia-smi
```

### Terminal 3: Monitor procesu (opcjonalnie)
```bash
watch -n 5 'ps aux | grep python | grep run.py'
```

## ⏱️ Oczekiwany czas

| Wersja | Dataset | Czas |
|--------|---------|------|
| `run.py` | 50k | **15-22h** |
| `run_ultra_safe.py` | 50k | **20-25h** |

## 📁 Wyjściowe pliki

```
./Mistral7b01-qlora-36h/          ← Model z run.py
./Mistral7b01-qlora-ultrasafe/    ← Model z run_ultra_safe.py
training_log_*.txt                 ← Logi (jeśli używasz run_safe.sh)
```

## 🔍 Sprawdzenie statusu w trakcie

```bash
# Sprawdź ostatnie linie logu
tail -f training_log_*.txt

# Sprawdź użycie GPU
nvidia-smi

# Sprawdź checkpointy
ls -lah ./Mistral7b01-qlora-36h/
```

## 🛑 Zatrzymanie treningu

```bash
# Gracefully (Ctrl+C w terminalu)
# Lub znajdź proces:
ps aux | grep python | grep run.py
kill -15 <PID>  # Graceful stop
```

## 🔄 Wznowienie z checkpointu

Jeśli trening został przerwany, możesz wznowić z ostatniego checkpointu.

Zmodyfikuj `run.py`:
```python
# Zamiast:
trainer.train()

# Użyj:
trainer.train(resume_from_checkpoint=True)
```

## ⚠️ Troubleshooting

### Problem: Out of Memory
```bash
# Użyj ultra-safe mode
.venv/bin/python run_ultra_safe.py

# Lub zmniejsz batch size w run.py
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
```

### Problem: Wolny trening
```bash
# Sprawdź czy GPU jest używane
nvidia-smi

# Sprawdź prędkość
# Powinno być ~2-3s per step dla run.py
# Jeśli wolniej, sprawdź:
# - Czy inne procesy nie używają GPU
# - Czy dataloader_num_workers nie jest za duży
```

### Problem: Brak miejsca na dysku
```bash
# Usuń stare checkpointy
rm -rf ./Mistral7b01-qlora-finetuned/checkpoint-*
rm -rf ./llama3-finedtuned-big5/

# Checkpointy zajmują ~500MB każdy
```

## 📦 Po zakończeniu

### Testowanie modelu
```python
from peft import AutoPeftModelForCausalLM
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    './Mistral7b01-qlora-36h',
    device_map='auto',
    dtype=torch.float16
)

# Użyj modelu...
```

### Merge LoRA adaptera z modelem bazowym (opcjonalnie)
```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained('./Mistral7b01-merged')
```

## 💾 Wymagania dyskowe

- Model bazowy: ~13GB
- Checkpoint: ~500MB
- Dataset cache: ~2GB
- **Łącznie**: ~20GB wolnego miejsca

---

## 🎯 Optymalne ustawienia dla Twojej karty

**RTX 5070 Ti 16GB** - `run.py`:
- ✅ Batch size: 4
- ✅ Gradient accumulation: 4
- ✅ Sequence length: 128
- ✅ Dataset: 50k przykładów
- ✅ Czas: ~15-22h
- ✅ VRAM peak: ~10-12GB
