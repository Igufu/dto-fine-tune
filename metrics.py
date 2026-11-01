import numpy as np
import evaluate

metric = evaluate.load("seqeval")

LABEL_NAMES = {
    "deontic": ["O", "B-DEO", "I-DEO"],
    "object": ["O", "B-DOBJ", "I-DOBJ"],
    "attribute": ["O", "B-ATTR", "I-ATTR"],
}


def make_compute_metrics(label_names):
    def compute_metrics(eval_preds):
        # usa logits como dijo claude, checkear si está bien
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [label_names[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    return compute_metrics


compute_metrics_fns = {
    name: make_compute_metrics(labels) for name, labels in LABEL_NAMES.items()
}


fake_labels = np.array([[1, 2, 0, -100]])
fake_logits = np.zeros((1, 4, 3))
fake_logits[..., 0] = 1.0
check = compute_metrics_fns["deontic"]((fake_logits, fake_labels))
print("sanity check (all-O prediction):", check)
assert check["f1"] == 0, "expected f1 == 0 for all-O prediction"
