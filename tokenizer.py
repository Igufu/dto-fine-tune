from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

NUM_LABELS = 3

raw_datasets = {
    'deontic': load_dataset('json', data_files='labeled_data/deontic.jsonl')['train'],
    'object': load_dataset('json', data_files='labeled_data/object.jsonl')['train'],
    'attribute': load_dataset('json', data_files='labeled_data/attribute.jsonl')['train'],
}

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/RoBERTalex", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    "PlanTL-GOB-ES/RoBERTalex", num_labels=NUM_LABELS
)

inputs = tokenizer(raw_datasets["deontic"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())


def shift_label(label):
    # If the label is B-XXX we change it to I-XXX
    if label % 2 == 1:
        label += 1
    return label


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            new_labels.append(shift_label(labels[word_id]))
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    new_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = {
    name: ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=ds.column_names,
    )
    for name, ds in raw_datasets.items()
}

'''
testeo, es bien raro el tokenizer del modelo
example = tokenized_datasets["deontic"][0]
tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
print(f'{"token":<20} {"label"}')
for tok, lab in zip(tokens, example["labels"]):
    print(f'{tok:<20} {lab}')
'''

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_datasets["deontic"][i] for i in range(2)])
print(batch["labels"])
for i in range(2):
    print(tokenized_datasets["deontic"][i]["labels"])