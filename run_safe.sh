#!/bin/bash

# Skrypt uruchamiający trening z monitoringiem VRAM
echo "🚀 Uruchamianie treningu z monitoringiem VRAM..."
echo "================================================"

# Ścieżka do interpretera Python
PYTHON="/home/bilus/PycharmProjects/Fine-Tuning-Pandora/.venv/bin/python"

# Wyczyść cache CUDA przed startem
$PYTHON -c "import torch; torch.cuda.empty_cache(); print('✅ CUDA cache wyczyszczony')"

# Ustaw zmienne środowiskowe dla stabilności
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Uruchom trening
$PYTHON run.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "✅ Trening zakończony. Log zapisany."
