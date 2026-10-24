from collections import defaultdict

with open('document.txt') as f:
    text = f.read()

anns = defaultdict(list)
with open('filtered.ann') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        info = parts[1].split(' ')
        cls, start, end = info[0], int(info[1]), int(info[2])
        anns[cls].append((start, end))


def tag(start, end, spans):
    for s, e in spans:
        if start >= s and end <= e:
            return 'B' if start == s else 'I'
    return 'O'


def build(spans):
    data = []
    offset = 0
    for line in text.split('\n'):
        tokens, tags = [], []
        i = 0
        while i < len(line):
            if line[i] == ' ':
                i += 1
                continue
            j = i
            while j < len(line) and line[j] != ' ':
                j += 1
            tokens.append(line[i:j])
            tags.append(tag(offset + i, offset + j, spans))
            i = j
        if any(t != 'O' for t in tags):
            data.append({'tokens': tokens, 'tags': tags})
        offset += len(line) + 1
    return data


for cls in ['Deontic', 'Attribute', 'DirectObject']:
    ds = build(anns[cls])
    example = next(d for d in ds if any(t == 'B' for t in d['tags']))
    print(f'=== {cls} ({len(ds)} examples) ===')
    for tok, tg in zip(example['tokens'], example['tags']):
        print(f'  {tok:<20} {tg}')
    print()
