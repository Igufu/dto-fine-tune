import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from tokenizer import tokenizer, tokenized_datasets
from metrics import LABEL_NAMES, compute_metrics_fns

MODEL_CHECKPOINT = "PlanTL-GOB-ES/RoBERTalex"

CLASS_WEIGHT_CAPS = {
    "deontic": None,
    "object": 2.5,
    "attribute": None,
}

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


for name, dataset in tokenized_datasets.items():
    split = dataset.train_test_split(test_size=0.2, seed=42)

    label_names = LABEL_NAMES[name]
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}

    counts = np.bincount(
        [l for ex in split["train"]["labels"] for l in ex if l != -100],
        minlength=3,
    )
    w = (counts.sum() / counts) ** 0.5
    class_weights = torch.tensor(w / w.min(), dtype=torch.float)
    cap = CLASS_WEIGHT_CAPS.get(name)
    if cap is not None:
        class_weights = torch.clamp(class_weights, max=cap)
    print(f"{name}: label counts={counts.tolist()}  weights={class_weights.tolist()}")

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

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fns[name],
        processing_class=tokenizer,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(f"models/{name}/final")
