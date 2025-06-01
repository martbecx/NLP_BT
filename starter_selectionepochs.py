# imports
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, FSMTForConditionalGeneration
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm

# function to tokenize dataset for translation

def preprocess_data(dataset_dict, tokenizer, src_lang, tgt_lang, split, max_length=128):
    """
    Preprocess translation datasets

    Args:
        dataset_dict: Dictionary containing train/dev/test datasets
        tokenizer: Tokenizer object
        src_lang: Source language code
        tgt_lang: Target language code
        split: Dataset split to preprocess ('train', 'validation', etc)
        max_length: Maximum sequence length
    Returns:
        tokenized_dataset: Preprocessed dataset for specified split
    """
    def preprocess_function(examples):
        inputs = examples[src_lang]
        targets = examples[tgt_lang]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[split].column_names
    )

    return tokenized_dataset

def postprocess_predictions(predictions, labels, tokenizer):
    """
    Convert model outputs to decoded text

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        tokenizer: Tokenizer object
    Returns:
        decoded_preds: Decoded predictions
        decoded_labels: Decoded labels
    """
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return decoded_preds, decoded_labels

# evaluation: for validation (with raw outputs) and testing (from text)

def compute_metrics_val(tokenizer, eval_preds):
    """
    Calculate BLEU score for predictions

    Args:
        tokenizer: Tokenizer object
        eval_preds: Tuple of predictions and labels
    Returns:
        metrics: Dictionary containing BLEU score
    """
    preds, labels = eval_preds
    decoded_preds, decoded_labels = postprocess_predictions(preds, labels, tokenizer)

    # Calculate BLEU score
    bleu = load("sacrebleu")
    results = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": results["score"]}

def compute_metrics_test(src, tgt, preds, bleu=True, comet=False):
    """
    Calculate BLEU score for predictions

    Args:
        src: Source language texts
        tgt: Target language texts
        preds: Predicted texts
        bleu: Whether to calculate BLEU score
        comet: Whether to calculate COMET score
    Returns:
        metrics: Dictionary containing BLEU score
    """
    if bleu:
        bleu = load("sacrebleu")
        results = bleu.compute(predictions=preds, references=[[l] for l in tgt])
        score = results["score"]
    if comet:
        comet = load("comet")
        results = comet.compute(predictions=preds, references=tgt, sources=src) 
        score = sum(results["scores"]) / len(results["scores"]) # avg comet score
    return score

# basic training loop for fine tuning

def train_model(model_name, tokenized_datasets, tokenizer, training_args):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Verify GPU usage
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"] if "dev" in tokenized_datasets else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics_val(tokenizer, x)
    )

    trainer.train()
    return model

# generation (on GPU) for test time
def translate_text(texts, model, tokenizer, max_length=128, batch_size=32):
    """
    Translate texts using the model

    Args:
        texts: List of texts to translate
        model: Translation model
        tokenizer: Tokenizer object
        max_length: Maximum sequence length
        batch_size: Batch size for translation
    Returns:
        translations: List of translated texts
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    translations = []

    # Create tqdm progress bar
    progress_bar = tqdm(range(0, len(texts), batch_size), desc="Translating")

    for i in progress_bar:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.0,
                early_stopping=True
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations


# Helper Fucntion
def generate_backtranslations(texts, model, tokenizer, max_length=128, batch_size=32):
    """
    Generate backtranslations from target language to source language.
    """
    return translate_text(texts, model, tokenizer, max_length=max_length, batch_size=batch_size)


def compute_cross_entropy(model, tokenizer, dataset):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ce_scores = []
    for sentence in tqdm(dataset['en'], desc="Computing cross-entropy"):
        # print(sentence)
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            ce_scores.append(loss.item())

    return ce_scores


def cross_entropy_difference(in_domain_model_name, out_domain_model_name='gpt2', train_set=None):
    """
    Calculate cross-entropy difference between two models for a list of sentences.
    """
    # Load pre-trained models
    in_domain_model = AutoModelForSeq2SeqLM.from_pretrained(in_domain_model_name)
    in_domain_tokenizer = AutoTokenizer.from_pretrained(in_domain_model_name)
    out_domain_model = AutoModelForCausalLM.from_pretrained(out_domain_model_name)
    out_domain_tokenizer = AutoTokenizer.from_pretrained(out_domain_model_name)

    if out_domain_tokenizer.pad_token is None:
        out_domain_tokenizer.add_special_tokens({"pad_token": "[pad]"})
        out_domain_model.resize_token_embeddings(len(out_domain_tokenizer))

    # Move models to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_domain_model = in_domain_model.to(device)
    out_domain_model = out_domain_model.to(device)

    # compute the CE of the dataset with each model
    ce_in_domain = compute_cross_entropy(in_domain_model, in_domain_tokenizer, train_set)
    ce_out_domain = compute_cross_entropy(out_domain_model, out_domain_tokenizer, train_set)

    # compute the differce in the CE scores
    ced_scores = [o - i for o, i in zip(ce_out_domain, ce_in_domain)]

    # Sort indices by CED
    sorted_indices = sorted(range(len(ced_scores)), key=lambda idx: ced_scores[idx])

    # Keep top-k most in-domain examples
    k = int(len(sorted_indices) * 0.9) # save top 90% 
    top_indices = sorted_indices[:k]

    # Filter dataset
    filtered_dataset = train_set.select(top_indices)

    return filtered_dataset


SRC_LANG = "en"
TGT_LANG = "ru"
# MODEL_NAME = f"facebook/wmt19-{SRC_LANG}-{TGT_LANG}"
# suggested model by seth 
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"
TRAIN_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
DEV_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
TEST_DATASET_NAME = "sethjsa/wmt20bio_ru_en_sent"
OUTPUT_DIR = "./results"

train_dataset = load_dataset(TRAIN_DATASET_NAME)
dev_dataset = load_dataset(DEV_DATASET_NAME)
test_dataset = load_dataset(TEST_DATASET_NAME)

USE_DATASELECTION = True

if USE_DATASELECTION:
    # data selection is used on the train set to improve the quality of the training data
    # datset has en and ru columns
    train_dataset['dev'] = cross_entropy_difference(MODEL_NAME, 'gpt2', train_dataset['dev'])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

USE_BACKTRANSLATION = True

if USE_BACKTRANSLATION:
    BT_MODEL_NAME = f"Helsinki-NLP/opus-mt-{TGT_LANG}-{SRC_LANG}"
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(BT_MODEL_NAME)
    bt_tokenizer = AutoTokenizer.from_pretrained(BT_MODEL_NAME)


training_data_dict = DatasetDict({"train": train_dataset["dev"]})

# change the splits for actual training. here, using flores-dev as training set because it's small (<1k examples)
tokenized_train_dataset = preprocess_data(training_data_dict, tokenizer, SRC_LANG, TGT_LANG, "train")
tokenized_dev_dataset = preprocess_data(dev_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")
tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, SRC_LANG, TGT_LANG, "test")

tokenized_datasets = DatasetDict({
    "train": tokenized_train_dataset,
    "dev": tokenized_dev_dataset,
    "test": tokenized_test_dataset
})


training_args = Seq2SeqTrainingArguments(
    torch_compile=False, # generally speeds up training, try without it to see if it's faster for small datasets
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32, # change batch sizes to fit your GPU memory and train faster
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    save_total_limit=1, # modify this to save more checkpoints
    num_train_epochs=1, # modify this to train more epochs
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=128,
    use_cpu=False,   # Set to False to enable GPU
    fp16=True,      # Enable mixed precision training for faster training
)




# BT + Original

# used_hashes = set()
# accumulated_src = []
# accumulated_tgt = []
# bt_model = bt_model.to("cuda" if torch.cuda.is_available() else "cpu")
# current_model_name = MODEL_NAME
# results_per_epoch = {}

# for epoch in range(1, 6):  # From epoch 1 to 5
#     print(f"\n=== Epoch {epoch}/5 ===")

#     # Select most in-domain subset from dev data using current model
#     selected_dataset = cross_entropy_difference(current_model_name, 'gpt2', train_dataset['dev'])

#     # Deduplicate and limit examples
#     new_examples = []
#     limit = len(selected_dataset) // 5
#     for ex in selected_dataset:
#         key = hash((ex[SRC_LANG], ex[TGT_LANG]))
#         if key not in used_hashes:
#             used_hashes.add(key)
#             new_examples.append(ex)
#         if len(new_examples) >= limit:
#             break

#     if not new_examples:
#         print("No more new examples to add.")
#         break

#     print(f"Adding {len(new_examples)} new examples to training via backtranslation.")

#     # Backtranslate: ru -> synthetic en
#     new_ru = [ex[TGT_LANG] for ex in new_examples]
#     synthetic_en = generate_backtranslations(new_ru, bt_model, bt_tokenizer)

#     # Update training data
#     accumulated_src.extend(synthetic_en)
#     accumulated_tgt.extend(new_ru)

#     combined_src = train_dataset["dev"][SRC_LANG] + accumulated_src
#     combined_tgt = train_dataset["dev"][TGT_LANG] + accumulated_tgt

#     combined_dataset = Dataset.from_dict({
#         SRC_LANG: combined_src,
#         TGT_LANG: combined_tgt
#     })
#     # combined_dataset = combined_dataset.shuffle(seed=epoch)
#     reduced_dataset = combined_dataset.select(range(int(len(combined_dataset) * 0.2)))

#     training_data_dict = DatasetDict({"train": reduced_dataset})
#     tokenized_train_dataset = preprocess_data(training_data_dict, tokenizer, SRC_LANG, TGT_LANG, "train")
#     tokenized_datasets["train"] = tokenized_train_dataset

#     # Update training arguments and directory
#     epoch_output_dir = f"{OUTPUT_DIR}/epoch_{epoch}"
#     training_args.output_dir = epoch_output_dir

#     # Train and save
#     model = train_model(current_model_name, tokenized_datasets, tokenizer, training_args)
#     model.save_pretrained(epoch_output_dir)
#     tokenizer.save_pretrained(epoch_output_dir)
#     current_model_name = epoch_output_dir

#     # Evaluate
#     predictions = translate_text(test_dataset["test"][SRC_LANG], model, tokenizer, max_length=128, batch_size=64)

#     comet_score = compute_metrics_test(
#         test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG],
#         predictions, bleu=False, comet=True
#     )
#     bleu_score = compute_metrics_test(
#         test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG],
#         predictions, bleu=True, comet=False
#     )

#     # Store results
#     results_per_epoch[epoch] = {
#         "comet": comet_score,
#         "bleu": bleu_score
#     }

#     print(f"Epoch {epoch} Results -> COMET: {comet_score:.2f}, BLEU: {bleu_score:.2f}")


# print("\n=== Summary of All Epochs ===")
# for epoch, scores in results_per_epoch.items():
#     print(f"Epoch {epoch}: COMET = {scores['comet']:.2f}, BLEU = {scores['bleu']:.2f}")


# results_original_only = {}

# original_dataset = train_dataset["dev"]
# slice_size = len(original_dataset) // 5

# current_model_name = MODEL_NAME  # Start with base model

# for epoch in range(1, 6):
#     print(f"\n=== Epoch {epoch}/5 — Original Only ===")

#     start_idx = (epoch - 1) * slice_size
#     end_idx = epoch * slice_size
#     epoch_slice = original_dataset.select(range(start_idx, end_idx))

#     # Prepare dataset
#     training_data_dict = DatasetDict({"train": epoch_slice})
#     tokenized_train_dataset = preprocess_data(training_data_dict, tokenizer, SRC_LANG, TGT_LANG, "train")
#     tokenized_datasets["train"] = tokenized_train_dataset

#     epoch_output_dir = f"{OUTPUT_DIR}/original_only_epoch_{epoch}"
#     training_args.output_dir = epoch_output_dir

#     model = train_model(current_model_name, tokenized_datasets, tokenizer, training_args)
#     model.save_pretrained(epoch_output_dir)
#     tokenizer.save_pretrained(epoch_output_dir)

#     current_model_name = epoch_output_dir

#     # Evaluate
#     predictions = translate_text(test_dataset["test"][SRC_LANG], model, tokenizer, max_length=128, batch_size=64)
#     comet_score = compute_metrics_test(test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG], predictions, bleu=False, comet=True)
#     bleu_score = compute_metrics_test(test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG], predictions, bleu=True, comet=False)

#     results_original_only[epoch] = {"comet": comet_score, "bleu": bleu_score}
#     print(f"Epoch {epoch} (Original Only) -> COMET: {comet_score:.2f}, BLEU: {bleu_score:.2f}")

# print("\n=== Original Only Results ===")
# for epoch, scores in results_original_only.items():
#     print(f"Epoch {epoch}: COMET = {scores['comet']:.2f}, BLEU = {scores['bleu']:.2f}")



results_bt_only = {}

# Generate BT data from dev
bt_model = bt_model.to("cuda" if torch.cuda.is_available() else "cpu")
selected_dataset = train_dataset["dev"]

# Backtranslate full dev set
bt_ru = selected_dataset[TGT_LANG]
bt_en = generate_backtranslations(bt_ru, bt_model, bt_tokenizer)

# Create a BT-only dataset
bt_dataset = Dataset.from_dict({
    SRC_LANG: bt_en,
    TGT_LANG: bt_ru
})

slice_size = len(bt_dataset) // 5
current_model_name = MODEL_NAME  # Start from base model

for epoch in range(1, 6):
    print(f"\n=== Epoch {epoch}/5 — BT Only ===")

    start_idx = (epoch - 1) * slice_size
    end_idx = epoch * slice_size
    epoch_slice = bt_dataset.select(range(start_idx, end_idx))

    # Prepare dataset
    training_data_dict = DatasetDict({"train": epoch_slice})
    tokenized_train_dataset = preprocess_data(training_data_dict, tokenizer, SRC_LANG, TGT_LANG, "train")
    tokenized_datasets["train"] = tokenized_train_dataset

    epoch_output_dir = f"{OUTPUT_DIR}/bt_only_epoch_{epoch}"
    training_args.output_dir = epoch_output_dir

    # Train on previous model
    model = train_model(current_model_name, tokenized_datasets, tokenizer, training_args)
    model.save_pretrained(epoch_output_dir)
    tokenizer.save_pretrained(epoch_output_dir)

    current_model_name = epoch_output_dir  # Set for next round

    # Evaluate
    predictions = translate_text(test_dataset["test"][SRC_LANG], model, tokenizer, max_length=128, batch_size=64)
    comet_score = compute_metrics_test(test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG], predictions, bleu=False, comet=True)
    bleu_score = compute_metrics_test(test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG], predictions, bleu=True, comet=False)

    results_bt_only[epoch] = {"comet": comet_score, "bleu": bleu_score}
    print(f"Epoch {epoch} (BT Only) -> COMET: {comet_score:.2f}, BLEU: {bleu_score:.2f}")

# Summary
print("\n=== BT Only Results ===")
for epoch, scores in results_bt_only.items():
    print(f"Epoch {epoch}: COMET = {scores['comet']:.2f}, BLEU = {scores['bleu']:.2f}")
