from datasets import load_dataset
import pandas as pd
import json

dataset = load_dataset("jingjietan/pandora-big5")
val_data = dataset['validation']  # NIE BYŁ w treningu!

print(f"✅ Validation set: {len(val_data)} przykładów")
print("💯 100% pewności że nie były w treningu")

# Konwertuj do DataFrame dla łatwiejszej analizy
df = pd.DataFrame({
    'text': [x['text'] for x in val_data],
    'O': [float(x['O']) for x in val_data],
    'C': [float(x['C']) for x in val_data],
    'E': [float(x['E']) for x in val_data],
    'A': [float(x['A']) for x in val_data],
    'N': [float(x['N']) for x in val_data],
    'ptype': [int(float(x['ptype'])) for x in val_data],
})

# Dodaj indeks oryginalny
df['original_index'] = df.index

# Teraz szukaj skrajnych przypadków
extreme_cases = []

for trait in ['O', 'C', 'E', 'A', 'N']:
    top5 = df.nlargest(5, trait).copy()
    top5['category'] = f'{trait}_highest'
    extreme_cases.extend(top5.to_dict('records'))
    
    bottom5 = df.nsmallest(5, trait).copy()
    bottom5['category'] = f'{trait}_lowest'
    extreme_cases.extend(bottom5.to_dict('records'))

# Deduplikacja po original_index
seen_indices = set()
unique_cases = []
for case in extreme_cases:
    if case['original_index'] not in seen_indices:
        seen_indices.add(case['original_index'])
        unique_cases.append(case)

print(f"Znaleziono {len(unique_cases)} unikalnych ekstremalnych przypadków")

# ==================== ZAPIS DO CSV ====================
test_df = pd.DataFrame(unique_cases)
test_df.to_csv('extreme_test_cases.csv', index=False, encoding='utf-8')
print(f"\n💾 Zapisano CSV: extreme_test_cases.csv")

# ==================== ZAPIS DO JSON ====================
with open('extreme_test_cases.json', 'w', encoding='utf-8') as f:
    json.dump(unique_cases, f, ensure_ascii=False, indent=2)
print(f"💾 Zapisano JSON: extreme_test_cases.json")

# ==================== ZAPIS GOTOWYCH PROMPTÓW ====================
with open('test_prompts.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EXTREME TEST CASES - VALIDATION SET\n")
    f.write(f"Total cases: {len(unique_cases)}\n")
    f.write("="*80 + "\n\n")
    
    for i, case in enumerate(unique_cases, 1):
        f.write(f"\n{'='*80}\n")
        f.write(f"TEST CASE #{i} - {case['category']}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Expected Output:\n")
        f.write(f"O:{int(case['O'])} C:{int(case['C'])} E:{int(case['E'])} A:{int(case['A'])} N:{int(case['N'])} Type:{case['ptype']}\n")
        f.write(f"\n{'─'*80}\n")
        f.write(f"Prompt for model:\n")
        f.write(f"{'─'*80}\n\n")
        
        f.write(f"### Instruction:\n")
        f.write(f"Analyze text and predict Big Five (0-100).\n\n")
        f.write(f"### Input:\n")
        f.write(f"{case['text']}\n\n")
        f.write(f"### Response:\n")
        f.write(f"[MODEL OUTPUT HERE]\n\n")

print(f"💾 Zapisano prompty: test_prompts.txt")

# ==================== ZAPIS SKRÓCONEJ WERSJI (tylko input/output) ====================
with open('test_cases_compact.txt', 'w', encoding='utf-8') as f:
    for i, case in enumerate(unique_cases, 1):
        f.write(f"# Case {i} ({case['category']})\n")
        f.write(f"INPUT: {case['text'][:200]}...\n")
        f.write(f"EXPECTED: O:{int(case['O'])} C:{int(case['C'])} E:{int(case['E'])} A:{int(case['A'])} N:{int(case['N'])} Type:{case['ptype']}\n")
        f.write(f"\n{'-'*80}\n\n")

print(f"💾 Zapisano wersję kompaktową: test_cases_compact.txt")

# ==================== STATYSTYKI ====================
print("\n📊 STATYSTYKI ZBIORU TESTOWEGO:")
print(f"\nLiczba przypadków: {len(unique_cases)}")
print("\nRozkład kategorii:")
categories = pd.Series([c['category'] for c in unique_cases]).value_counts()
print(categories)

print("\n📊 Statystyki wartości:")
for trait in ['O', 'C', 'E', 'A', 'N']:
    print(f"\n{trait}:")
    print(f"  Min: {test_df[trait].min():.1f}")
    print(f"  Max: {test_df[trait].max():.1f}")
    print(f"  Mean: {test_df[trait].mean():.1f}")
    print(f"  Std: {test_df[trait].std():.1f}")

# ==================== PODGLĄD PRZYKŁADÓW ====================
print("\n" + "="*80)
print("📋 PRZYKŁADOWE PRZYPADKI (pierwsze 3):")
print("="*80)

for i, case in enumerate(unique_cases[:3], 1):
    print(f"\n{i}. {case['category']}")
    print(f"   O:{int(case['O'])} C:{int(case['C'])} E:{int(case['E'])} A:{int(case['A'])} N:{int(case['N'])} Type:{case['ptype']}")
    print(f"   Text: {case['text'][:150]}...")

print("\n✅ WSZYSTKO ZAPISANE!")
print("\nPliki:")
print("  📄 extreme_test_cases.csv - pełne dane w CSV")
print("  📄 extreme_test_cases.json - pełne dane w JSON")
print("  📄 test_prompts.txt - gotowe prompty do testowania")
print("  📄 test_cases_compact.txt - skrócona wersja do przeglądu")