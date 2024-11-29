from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# loading model
model= AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")


tokenizer = AutoTokenizer.from_pretrained(model_name)

# loading dataset for finetune model
datasets = load_dataset("stanfordnlp/imdb") 

datasets =datasets.map(lambda row: tokenizer(row["text"],padding = True,max_length=512, truncation=True),keep_in_memory=True,batched=True)


print("Number of GPUs:", torch.cuda.device_count())



training_args= TrainingArguments(output_dir="./results",per_device_train_batch_size=32,
                                # per_gpu_train_batch_size=32,
                                gradient_accumulation_steps=2, 
                                # save_strategy="epoch",    # Save model every epoch
                                # evaluation_strategy="epoch",
                                # load_best_model_at_end=True, 
                                fp16=True,          # Enable mixed precision for faster training
                                dataloader_num_workers=4
                                )
# datacollectorwithpadding is a utility provided by hugging face that ensure all imput data has same length during traiing
from transformers import DataCollatorWithPadding

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()

# /kaggle/fine_tuned_model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


import shutil

# Compress the model directory into a zip file
shutil.make_archive("fine_tuned_model", "zip", "./fine_tuned_model")

from IPython.display import FileLink

# Generate a download link
FileLink(r'fine_tuned_model.zip')