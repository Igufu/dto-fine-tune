import string

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

INPUT_PATH = "input/chap_1_&_2.txt"
OUTPUT_PATH = "predictions.ann"

PUNCT = set(string.punctuation + "¿¡«»")
STOPWORDS = {"de", "la", "el", "los", "las", "y", "a", "en", "del", "que", "por", "con"}


def clean_span(start, end, doc):
    while start < end and (doc[start] in PUNCT or doc[start].isspace()):
        start += 1
    while end > start and (doc[end - 1] in PUNCT or doc[end - 1].isspace()):
        end -= 1
    return start, end


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
        aggregation_strategy="first",
    )

    offset = 0
    for line in text.split("\n"):
        if line.strip():
            for e in nlp(line):
                start, end = clean_span(
                    offset + int(e["start"]), offset + int(e["end"]), text
                )
                if start >= end:
                    continue
                body = text[start:end]
                words = [w for w in body.split() if w]
                if not words or all(w.lower() in STOPWORDS for w in words):
                    continue
                spans.append((cls_label, start, end, body))
        offset += len(line) + 1

with open(OUTPUT_PATH, "w") as f:
    for i, (cls, start, end, span_text) in enumerate(spans, 1):
        f.write(f"T{i}\t{cls} {start} {end}\t{span_text}\n")

print(f"wrote {len(spans)} spans -> {OUTPUT_PATH}")
for cls, start, end, span_text in spans[:20]:
    print(f"  {cls:<12} {start:>7}-{end:<7} {span_text}")
