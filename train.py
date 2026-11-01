from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from tokenizer import tokenizer, tokenized_datasets
from metrics import LABEL_NAMES, compute_metrics_fns

MODEL_CHECKPOINT = "PlanTL-GOB-ES/RoBERTalex"

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

for name, dataset in tokenized_datasets.items():
    split = dataset.train_test_split(test_size=0.2, seed=42)

    label_names = LABEL_NAMES[name]
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"models/{name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fns[name],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"models/{name}/final")
