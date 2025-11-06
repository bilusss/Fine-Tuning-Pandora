import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === ŚCIEŻKA DO ZAPISANEGO MODELU ===
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
FINETUNED_MODEL_DIR = "./LLama3-qlora-ultrasafe-150k"  # katalog z wynikami treningu

# === ŁADOWANIE TOKENIZERA ===
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# === ŁADOWANIE MODELU BAZOWEGO W 4-BIT ===
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# === NAŁOŻENIE WYSZKOLONYCH WAG LoRA ===
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)
model.eval()

# === TEST PROMPT ===
prompt = """### Instruction:
Analyze text and predict Big Five (0-100).

### Input:
lol this wasn't directed towards you. this was a joke that would be found offensive pretty much universally.I will message it to you privately.

### Response:
O:<int> C:<int> E:<int> A:<int> N:<int> Type:<int>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n===== WYNIK =====")
print(tokenizer.decode(output[0], skip_special_tokens=True))
