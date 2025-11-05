# ==================== ULTRA-SAFE VERSION DLA RTX 5070 Ti 16GB ====================
# Ta wersja ma NAJBEZPIECZNIEJSZE ustawienia - gwarantowane działanie bez OOM

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

# ==================== ULTRA-SAFE CONFIG ====================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "jingjietan/pandora-big5"
OUTPUT_DIR = "./Mistral7b01-qlora-ultrasafe"

# ULTRA-SAFE: Najmniejsze możliwe wartości dla gwarancji działania
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

BATCH_SIZE = 2  # Minimalny batch
GRADIENT_ACCUMULATION_STEPS = 8  # Efektywny batch = 16
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 128

# 50k przykładów, efektywny batch=16 → 3125 effective steps
# ~3s/krok × 8 accum = ~24s/effective step
# 3125 × 24s = 20.8h
TRAIN_SIZE = 50000
VAL_SIZE = 5000

print("="*70)
print("🛡️  ULTRA-SAFE MODE - GWARANTOWANE DZIAŁANIE NA 16GB")
print("="*70)
print(f"Batch size: {BATCH_SIZE} (efektywny: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"Max seq length: {MAX_SEQ_LENGTH}")
print(f"Dataset: {TRAIN_SIZE} train, {VAL_SIZE} validation")
print(f"Estimated time: ~20-25h")
print("="*70)

# Quantization
print("\n🔧 Konfiguracja 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
print("📥 Ładowanie modelu...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

print(f"✅ Model załadowany! VRAM: ~{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# LoRA
print("🎯 Konfiguracja LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dataset
print(f"\n📚 Ładowanie datasetu...")
dataset = load_dataset(DATASET_NAME)
print(f"Original - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Preprocessing
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
        
        prompt = f"""### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
{user_text}

### Response:
O:{o_score} C:{c_score} E:{e_score} A:{a_score} N:{n_score} Type:{ptype}"""
        texts.append(prompt)
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,
    )

# Sample dataset
print(f"\n🔄 Przygotowanie datasetu ({TRAIN_SIZE//1000}k train, {VAL_SIZE//1000}k val)...")
import random
random.seed(42)

train_indices = random.sample(range(len(dataset['train'])), min(TRAIN_SIZE, len(dataset['train'])))
val_indices = random.sample(range(len(dataset['validation'])), min(VAL_SIZE, len(dataset['validation'])))

dataset['train'] = dataset['train'].select(train_indices)
dataset['validation'] = dataset['validation'].select(val_indices)

print(f"✅ Dataset: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")

# Tokenize
print("\n🔄 Tokenizacja...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=500,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizacja",
    num_proc=2,  # Ultra-safe: tylko 2 procesy
)

print(f"✅ Tokenizacja zakończona!")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments - ULTRA-SAFE
print("\n⚙️ Konfiguracja treningu (ultra-safe)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    fp16=True,
    bf16=False,
    tf32=False,
    max_grad_norm=0.3,
    weight_decay=0.001,
    
    report_to="none",
    ddp_find_unused_parameters=False,
    
    # ULTRA-SAFE dataloader settings
    dataloader_num_workers=0,  # Brak workerów - najprostsze
    dataloader_pin_memory=False,  # Wyłączone dla bezpieczeństwa
    group_by_length=False,
    
    torch_compile=False,
)

# Trainer
print("\n🚀 Inicjalizacja Trainera...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Stats
steps = len(tokenized_dataset['train']) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
print(f"\n📊 Effective steps: {steps}")
print(f"📊 GPU steps: {steps * GRADIENT_ACCUMULATION_STEPS}")
print(f"📊 Estimated time: {steps * GRADIENT_ACCUMULATION_STEPS * 3 / 3600:.1f}h - {steps * GRADIENT_ACCUMULATION_STEPS * 4 / 3600:.1f}h")

# Train
print("\n🎓 Rozpoczynam trening (ultra-safe mode)...")
torch.cuda.empty_cache()

import time
start_time = time.time()

trainer.train()

end_time = time.time()
total_time = (end_time - start_time) / 3600

print(f"\n✅ Trening zakończony w {total_time:.2f}h!")

# Save
print("\n💾 Zapisywanie modelu...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model zapisany w: {OUTPUT_DIR}")

# Eval
print("\n📊 Finalna ewaluacja...")
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

# Test
print("\n🧪 Test generowania...")
model.eval()

test_prompt = """### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
I love meeting new people and exploring creative ideas!

### Response:
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n{generated}")

# Stats
print("\n📈 Statystyki VRAM:")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n🎉 Trening zakończony!")
print(f"Czas: {total_time:.2f}h / 36h ({total_time / 36 * 100:.1f}%)")
