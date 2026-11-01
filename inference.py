from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

MODELS = {
    "deontic": ("models/deontic/final", "Deontic"),
    "object": ("models/object/final", "DirectObject"),
    "attribute": ("models/attribute/final", "Attribute"),
}

INPUT_PATH = "input/document.txt"
OUTPUT_PATH = "predictions.ann"

with open(INPUT_PATH) as f:
    text = f.read()

spans = []
for name, (path, cls_label) in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(path)
    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    offset = 0
    for line in text.split("\n"):
        if line.strip():
            for e in nlp(line):
                start = offset + int(e["start"])
                end = offset + int(e["end"])
                spans.append((cls_label, start, end, text[start:end]))
        offset += len(line) + 1

with open(OUTPUT_PATH, "w") as f:
    for i, (cls, start, end, span_text) in enumerate(spans, 1):
        f.write(f"T{i}\t{cls} {start} {end}\t{span_text}\n")

print(f"wrote {len(spans)} spans -> {OUTPUT_PATH}")
for cls, start, end, span_text in spans[:20]:
    print(f"  {cls:<12} {start:>7}-{end:<7} {span_text}")
