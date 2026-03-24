import git
from huggingface_hub import login
from .data import get_dataset_cls
from .metrics import get_metric_cls
from .ld import build_airo_model_insert_query
from ..config import get_config

from helpers import update
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer


def train(
        decisions: list[dict[str, str | list[str]]],
        labels: list[str],
        model_id: str,
        transformer: str = "distilbert/distilbert-base-uncased",
        learning_rate: float = 2e-5,
        epochs: int = 2,
        weight_decay: float = 0.01
):
    multi_labeled = False
    for decision in decisions:
        if len(decision["classes"]) > 1:
            multi_labeled = True
            break
    
    if multi_labeled:
        problem_type = "multi_label_classification"
    else:
        problem_type = "single_label_classification"

    config = get_config()
    token = config.ml_training.huggingface_token
    if not token:
        raise RuntimeError(
            "No HuggingFace token configured in config.json (ml_training.huggingface_token)"
        )
    login(token=token.get_secret_value())

    # First load utility classes for data and metrics
    dataset = get_dataset_cls(problem_type)(decisions, labels)
    metrics = get_metric_cls(problem_type)

    # Then load tokenizer, model, data collator, training arguments, and trainer
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    tokenized_data = dataset.format().map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_id,
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,  # we'll push manually at the end to grab the commitinfo
        push_to_hub_model_id=model_id
    )

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        transformer,
        num_labels=len(dataset.id2label),
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        problem_type=problem_type
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics().compute,
    )

    # Train
    trainer.train()

    # Push best model to hub and evaluate metrics
    commit_info = None
    try:
        commit_info = trainer.push_to_hub(blocking=True)
    except Exception as exc:
        print(f"Push to hub skipped/failed: {exc}", flush=True)
    results = trainer.evaluate()

    repo = git.Repo(search_parent_directories=True)

    if commit_info:
        # Build SPARQL INSERT query
        query_str = build_airo_model_insert_query(
            hub_model_id=model_id,
            commit_oid=commit_info.oid,
            code_git_sha=repo.head.object.hexsha,
            hf_repo_url=commit_info.repo_url.url,
            hf_tree_url=f"{commit_info.repo_url.url}/tree/main/",
            source_repo_url=repo.remote().url,
            results=results
        )

        update(query_str, sudo=True)

        print(query_str, flush=True)
