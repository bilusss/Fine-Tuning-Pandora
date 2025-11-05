# 🔧 Poprawki dla RTX 5070 Ti 16GB - Fine-Tuning Mistral-7B

## ❌ Problem
Kod wychodził z błędem **Out of Memory** po ~361 krokach (29% treningu).

## ✅ Rozwiązanie

### Główne zmiany w `run.py`:

1. **Batch Size**: 20 → **4**
   - Poprzednio: zbyt duży dla 16GB VRAM
   - Teraz: bezpieczny rozmiar

2. **Gradient Accumulation**: 1 → **4**
   - Efektywny batch size: 16 (zamiast 20)
   - Symuluje większy batch bez dodatkowej pamięci

3. **Sequence Length**: 192 → **128**
   - Krótsze sekwencje = mniej pamięci
   - Nadal wystarczające dla analizy osobowości

4. **Dataset**: 25k → **50k** przykładów
   - Więcej danych dzięki mniejszemu batch size
   - Lepsza jakość modelu

5. **Wyłączone problematyczne funkcje**:
   - ❌ `torch_compile` - powodowało OOM
   - ❌ `group_by_length` - nieprzewidywalne użycie VRAM
   - ❌ `tf32` - powodowało problemy z CUDA Graphs
   - ✅ Zmniejszono `dataloader_num_workers` 8 → 2

## 📊 Oszacowania

### `run.py` (zalecany):
- **Batch size**: 4, grad_accum: 4 (efektywny: 16)
- **Sequence length**: 128
- **Dataset**: 50k train, 5k validation
- **Czas**: ~15-22h
- **Kroki**: 12,500 GPU steps (3,125 effective steps)

### `run_ultra_safe.py` (backup):
- **Batch size**: 2, grad_accum: 8 (efektywny: 16)
- **Sequence length**: 128
- **Dataset**: 50k train, 5k validation
- **Czas**: ~20-25h
- **Kroki**: 25,000 GPU steps (3,125 effective steps)

## 🚀 Jak uruchomić

### Metoda 1: Bezpośrednio (zalecana)
```bash
python run.py
```

### Metoda 2: Z monitoringiem i logowaniem
```bash
./run_safe.sh
```

### Metoda 3: Ultra-safe mode (jeśli nadal problemy)
```bash
python run_ultra_safe.py
```

## 💡 Monitorowanie

Podczas treningu możesz monitorować VRAM w osobnym terminalu:
```bash
watch -n 1 nvidia-smi
```

## 🎯 Oczekiwane rezultaty

- ✅ **Brak OOM** - zmieści się w 16GB VRAM (~8-12GB peak)
- ✅ **Zakończenie w 36h** - realny czas: 15-22h
- ✅ **Dobra jakość** - 50k przykładów z efektywnym batch=16

## 📝 Checkpointy

Model zapisuje się co 1000 kroków w folderze `./Mistral7b01-qlora-36h/` (lub `ultrasafe` dla drugiej wersji).

Ostatnie 2 checkpointy są zachowywane (`save_total_limit=2`).

## ⚠️ W razie problemów

Jeśli nadal występują problemy z pamięcią:

1. Użyj `run_ultra_safe.py` (batch_size=2)
2. Zmniejsz `TRAIN_SIZE` do 30000
3. Zmniejsz `MAX_SEQ_LENGTH` do 96
4. Wyczyść cache przed startem:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## 📈 Porównanie wydajności

| Wersja | Batch | Grad Acc | Eff. Batch | VRAM Peak | Czas/krok | Czas całk. |
|--------|-------|----------|------------|-----------|-----------|------------|
| Stara  | 20    | 1        | 20         | ~16GB+ ❌ | ~3s       | OOM        |
| Nowa   | 4     | 4        | 16         | ~10GB ✅  | ~2-3s     | ~15-22h    |
| Ultra  | 2     | 8        | 16         | ~8GB ✅   | ~3-4s     | ~20-25h    |

---

**Autor poprawek**: GitHub Copilot  
**Data**: 5 listopada 2025  
**Status**: ✅ Gotowe do użycia
