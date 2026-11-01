import json
import string
from collections import defaultdict

INPUT_DIR = 'input'
OUTPUT_DIR = 'labeled_data'
STANDOFF_DOCUMENT = f'{INPUT_DIR}/standoff_comp.ann'

PUNCT = set(string.punctuation + '¿¡«»')

with open(f'{INPUT_DIR}/document.txt') as f:
    text = f.read()

anns = defaultdict(list)
with open(STANDOFF_DOCUMENT) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        info = parts[1].split(' ')
        cls, start, end = info[0], int(info[1]), int(info[2])
        while start < end and text[start] in ' \n':
            start += 1
        while end > start and text[end - 1] in ' \n':
            end -= 1
        anns[cls].append((start, end))


def tag_id(start, end, spans):
    for s, e in spans:
        if s <= start < e:
            return 1 if start == s else 2
    return 0


def split_word(word_start, word):
    """Split leading/trailing punctuation. Returns [(abs_start, text, is_core)]."""
    a, b = 0, len(word)
    while a < b and word[a] in PUNCT:
        a += 1
    while b > a and word[b - 1] in PUNCT:
        b -= 1
    pieces = []
    for k in range(a):
        pieces.append((word_start + k, word[k], False))
    if a < b:
        pieces.append((word_start + a, word[a:b], True))
    for k in range(b, len(word)):
        pieces.append((word_start + k, word[k], False))
    return pieces


def build(spans):
    data = []
    offset = 0
    for line in text.split('\n'):
        tokens, ner_tags = [], []
        i = 0
        while i < len(line):
            if line[i] == ' ':
                i += 1
                continue
            j = i
            while j < len(line) and line[j] != ' ':
                j += 1
            raw = line[i:j]
            orig_tag = tag_id(offset + i, offset + j, spans)
            for abs_start, piece_text, is_core in split_word(offset + i, raw):
                tokens.append(piece_text)
                if not is_core:
                    ner_tags.append(0)
                elif orig_tag == 1:
                    ner_tags.append(1)
                else:
                    ner_tags.append(tag_id(abs_start, abs_start + len(piece_text), spans))
            i = j
        if any(t != 0 for t in ner_tags):
            data.append({'tokens': tokens, 'ner_tags': ner_tags})
        offset += len(line) + 1
    return data


datasets = {
    'deontic': ('Deontic', ['O', 'B-DEO', 'I-DEO']),
    'object': ('DirectObject', ['O', 'B-DOBJ', 'I-DOBJ']),
    'attribute': ('Attribute', ['O', 'B-ATTR', 'I-ATTR']),
}

for name, (cls, label_names) in datasets.items():
    ds = build(anns[cls])
    with open(f'{OUTPUT_DIR}/{name}.jsonl', 'w') as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    example = next(d for d in ds if any(t != 0 for t in d['ner_tags']))
    print(f'=== {name} ({len(ds)} examples) -> {OUTPUT_DIR}/{name}.jsonl ===')
    print(f'  labels: {label_names}')
    for tok, tg in zip(example['tokens'], example['ner_tags']):
        print(f'  {tok:<20} {tg} ({label_names[tg]})')
    print()
