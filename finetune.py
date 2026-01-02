from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor
from config import BASE_MODEL, BATCH_SIZE, EPOCHS, GRAD_ACCUMULATION, LEARNING_RATE, MAX_SEQ_LENGTH, OUTPUT_FILE
from prepare_data import get_datasets

# downloading model and processor
try:
    model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    trust_remote_code=True,
    use_gradient_checkpointing=True
)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# define model
model = FastVisionModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

# get train and eval datasets
train_dataset, eval_dataset = get_datasets()

# define data collator
data_collator = UnslothVisionDataCollator(model, tokenizer)

# define trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
        report_to="none",
        save_total_limit=2,
    ),
)

# fine-tuning
try:
    trainer_stats = trainer.train()
    print("Training completed.")
    print(trainer_stats)    
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# saving the fine-tuned model
try:
    print("Saving model...")
    model.save_pretrained("oi_analyst_model")
    tokenizer.save_pretrained("oi_analyst_model")

    print("\n=== Training Complete ===")
    print(f"Total training time: {trainer_stats.metrics.get('train_runtime', 0):.2f}s")
    print(f"Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")
    print(f"Model saved to: oi_analyst_model/")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)
