# ==================== IMPORTY ====================
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import re
import emoji

def clean_text_for_personality(text):
    """Czyści tekst zachowując elementy istotne dla analizy osobowości"""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' {3,}', ' ', text)
    text = text.strip()
    
    if len(text.strip()) < 3:
        return "[cleaned empty text]"
    
    return text

# ==================== KONFIGURACJA DLA RTX 5070 Ti (16GB) ====================
# Cel: Maksymalnie 36 godzin (1.5 dnia) treningu
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "jingjietan/pandora-big5"
OUTPUT_DIR = "./Mistral7b01-qlora-36h"

# LoRA parametry - zoptymalizowane dla pamięci
LORA_R = 8  # Niski rank dla szybkości
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training parametry - BEZPIECZNE DLA 16GB VRAM
BATCH_SIZE = 4  # Bezpieczny dla 16GB
GRADIENT_ACCUMULATION_STEPS = 4  # Efektywny batch = 16
NUM_EPOCHS = 1  # Tylko 1 epoka
LEARNING_RATE = 2e-4  # Stabilny learning rate
MAX_SEQ_LENGTH = 128  # Zmniejszone dla oszczędności pamięci

# Dataset - zoptymalizowany dla 36h z batch=4
# ~1.5-2s/krok × 4 batch × 4 grad_accum = ~10-15s/effective step
# 36h = 129,600s → ~10,000 effective steps możliwe
# 10,000 steps × 16 effective_batch = ~160,000 przykładów max
TRAIN_SIZE = 50000  # 50k przykładów - więcej danych przez mniejszy batch
VAL_SIZE = 5000     # 5k walidacji

print("="*70)
print("🚀 KONFIGURACJA DLA RTX 5070 Ti 16GB - MAKSYMALNIE 36H")
print("="*70)
print(f"GPU: RTX 5070 Ti (352 TFLOPS FP16, 16GB VRAM)")
print(f"Dataset: {TRAIN_SIZE} train, {VAL_SIZE} validation")
print(f"Batch size: {BATCH_SIZE} (efektywny: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"Max seq length: {MAX_SEQ_LENGTH}")
print(f"LoRA rank: {LORA_R}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Estimated steps: {TRAIN_SIZE // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
print(f"Expected time/step: ~10-15s (effective step)")
print(f"Total estimated time: {(TRAIN_SIZE // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * 15 / 3600:.1f}h")
print("="*70)

# ==================== QUANTIZATION CONFIG ====================
print("\n🔧 Konfiguracja 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # FP16 dla RTX 5070 Ti!
    bnb_4bit_use_double_quant=True,
)

# ==================== ZAŁADUJ MODEL I TOKENIZER ====================
print("📥 Ładowanie modelu i tokenizera...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,  # FP16 dla RTX 5070 Ti
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

print(f"✅ Model załadowany! VRAM użyty: ~{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==================== KONFIGURACJA LORA ====================
print("🎯 Konfiguracja LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== ZAŁADUJ DATASET ====================
print(f"\n📚 Ładowanie datasetu: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME)

print(f"Original - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# ==================== PREPROCESSING ====================
def preprocess_function(examples):
    texts = []
    
    for i in range(len(examples['text'])):
        o_score = int(float(examples['O'][i]))
        c_score = int(float(examples['C'][i]))
        e_score = int(float(examples['E'][i]))
        a_score = int(float(examples['A'][i]))
        n_score = int(float(examples['N'][i]))
        ptype = int(float(examples['ptype'][i]))
        
        user_text = clean_text_for_personality(examples['text'][i])
        
        if not user_text or user_text == "[cleaned empty text]":
            user_text = "[Empty text]"
        
        # Zoptymalizowany format
        prompt = f"""### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
{user_text}

### Response:
O:{o_score} C:{c_score} E:{e_score} A:{a_score} N:{n_score} Type:{ptype}"""
        
        texts.append(prompt)
    
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,
    )
    
    return model_inputs

# ==================== ZMNIEJSZ DATASET ====================
print(f"\n🔄 Przygotowanie datasetu ({TRAIN_SIZE//1000}k train, {VAL_SIZE//1000}k val)...")

import random
random.seed(42)

# Losowe próbkowanie - prostsza metoda
train_indices = random.sample(range(len(dataset['train'])), min(TRAIN_SIZE, len(dataset['train'])))
val_indices = random.sample(range(len(dataset['validation'])), min(VAL_SIZE, len(dataset['validation'])))

dataset['train'] = dataset['train'].select(train_indices)
dataset['validation'] = dataset['validation'].select(val_indices)

print(f"✅ Dataset przygotowany: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")

# Przykłady
print("\n📋 Przykłady (pierwsze 2):")
for idx in range(min(2, len(dataset['train']))):
    sample = dataset['train'][idx]
    original = sample['text']
    cleaned = clean_text_for_personality(original)
    print(f"\n[{idx+1}] ({len(cleaned)} znaków)")
    print(f"    Tekst: {cleaned[:100]}...")

# ==================== TOKENIZACJA ====================
print("\n🔄 Tokenizacja (z wielowątkowością)...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizacja",
    num_proc=4,  # Zmniejszone dla stabilności
)

print(f"✅ Tokenizacja zakończona!")
print(f"Train: {len(tokenized_dataset['train'])}, Val: {len(tokenized_dataset['validation'])}")

# Sprawdź rozkład długości
lengths = [len(ex['input_ids']) for ex in tokenized_dataset['train']]
print(f"\nStatystyki długości sekwencji:")
print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Średnia: {sum(lengths)/len(lengths):.1f}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==================== TRAINING ARGUMENTS - ZOPTYMALIZOWANE DLA 16GB ====================
print("\n⚙️ Konfiguracja treningu (stabilna dla 16GB VRAM)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    
    # ===== BEZPIECZNE BATCH SIZES =====
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Większy dla eval (bez gradientów)
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # ===== MINIMALNE CHECKPOINTY =====
    logging_steps=50,  # Loguj co 50 kroków
    save_strategy="steps",
    save_steps=1000,  # Zapisuj co 1000 kroków
    eval_strategy="steps",
    eval_steps=1000,  # Ewaluuj co 1000 kroków
    save_total_limit=2,
    
    # ===== OPTYMALIZACJE =====
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",  # 8-bit optimizer dla oszczędności VRAM
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    fp16=True,  # FP16 dla RTX 5070 Ti
    bf16=False,
    tf32=False,  # WYŁĄCZONE - powodowało problemy
    max_grad_norm=0.3,
    weight_decay=0.001,
    
    report_to="none",
    ddp_find_unused_parameters=False,
    
    # ===== DATALOADER - KONSERWATYWNE USTAWIENIA =====
    dataloader_num_workers=2,  # Zmniejszone dla stabilności
    dataloader_pin_memory=True,
    dataloader_persistent_workers=False,  # Wyłączone dla oszczędności pamięci
    group_by_length=False,  # WYŁĄCZONE - powoduje wahania VRAM
    
    # ===== EKSPERYMENTALNE - WYŁĄCZONE =====
    torch_compile=False,  # Wyłączone - powodowało OOM
)

# ==================== TRAINER ====================
print("\n🚀 Inicjalizacja Trainera...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# ==================== SZCZEGÓŁOWE OSZACOWANIE CZASU ====================
steps_per_epoch = len(tokenized_dataset['train']) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
num_evals = steps_per_epoch // 1000  # Co 1000 kroków

print("\n" + "="*70)
print("📊 SZCZEGÓŁOWE OSZACOWANIE CZASU (RTX 5070 Ti 16GB):")
print("="*70)
print(f"Całkowita liczba kroków (effective): {steps_per_epoch}")
print(f"Całkowita liczba kroków (GPU): {steps_per_epoch * GRADIENT_ACCUMULATION_STEPS}")
print(f"Liczba ewaluacji: {num_evals}")
print(f"")
print(f"Bazując na RTX 5070 Ti (352 TFLOPS FP16):")
print(f"  Szacowany czas/GPU krok (FP16, batch=4, seq=128): ~2-3s")
print(f"  Czas treningu: {steps_per_epoch * GRADIENT_ACCUMULATION_STEPS} kroków × 2.5s = {steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 2.5 / 3600:.1f}h")
print(f"  Czas ewaluacji: {num_evals} × 3 min = {num_evals * 3 / 60:.1f}h")
print(f"")
print(f"🎯 CAŁKOWITY CZAS: ~{(steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 2.5 / 3600) + (num_evals * 3 / 60):.1f}h")
print(f"")
print(f"📌 Pesymistyczny: {(steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 4 / 3600) + (num_evals * 5 / 60):.1f}h")
print(f"📌 Realistyczny: {(steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 2.5 / 3600) + (num_evals * 3 / 60):.1f}h")
print(f"📌 Optymistyczny: {(steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 2 / 3600) + (num_evals * 2 / 60):.1f}h")
print("="*70)

realistic_hours = (steps_per_epoch * GRADIENT_ACCUMULATION_STEPS * 2.5 / 3600) + (num_evals * 3 / 60)
if realistic_hours > 36:
    print(f"⚠️  UWAGA: Może przekroczyć 36h! Rozważ zmniejszenie do {int(TRAIN_SIZE * 36 / realistic_hours)}k przykładów")
else:
    print(f"✅ Powinno zmieścić się w 36h z zapasem!")

# ==================== ROZPOCZNIJ TRENING ====================
print("\n🎓 Rozpoczynam trening (max 36h)...")
print(f"Efektywny batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"⚠️  Użycie VRAM będzie monitorowane...")

torch.cuda.empty_cache()

# TRENING!
import time
start_time = time.time()

trainer.train()

end_time = time.time()
total_time = (end_time - start_time) / 3600

print(f"\n✅ Trening zakończony w {total_time:.2f}h!")

# ==================== ZAPISZ MODEL ====================
print("\n💾 Zapisywanie modelu...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model zapisany w: {OUTPUT_DIR}")

# ==================== EWALUACJA ====================
print("\n📊 Finalna ewaluacja...")
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

# ==================== QUICK TEST ====================
print("\n🧪 Test generowania...")

def generate_text(prompt, max_new_tokens=100):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = """### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
I love meeting new people and exploring creative ideas! Sometimes I'm disorganized but that's okay.

### Response:
"""

generated = generate_text(test_prompt)
print(f"\n{generated}")

# Test z prawdziwym przykładem
sample = dataset['validation'][0]
test_real = f"""### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
{clean_text_for_personality(sample['text'])}

### Response:
"""

generated_real = generate_text(test_real)
print(f"\n\nTest z validation set:")
print(f"Prawdziwe: O:{sample['O']} C:{sample['C']} E:{sample['E']} A:{sample['A']} N:{sample['N']}")
print(f"Predykcja:\n{generated_real}")

# ==================== STATYSTYKI ====================
print("\n📈 Statystyki VRAM:")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n🎉 Trening zakończony!")
print(f"Czas treningu: {total_time:.2f}h / 36h")
print(f"Wykorzystanie limitu: {total_time / 36 * 100:.1f}%")

print(f"\nAby załadować model:")
print(f"""
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    '{OUTPUT_DIR}',
    device_map='auto',
    torch_dtype=torch.float16
)
""")