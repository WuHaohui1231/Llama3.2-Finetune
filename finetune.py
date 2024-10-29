from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# # plt.style.use('ggplot')

# import torch
# from trl import SFTTrainer
# from transformers import TrainingArguments, TextStreamer
# from unsloth.chat_templates import get_chat_template
# from unsloth import FastLanguageModel
# from datasets import Dataset
# from unsloth import is_bfloat16_supported

# # Saving model
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Warnings
# # import warnings
# # warnings.filterwarnings("ignore")




# data = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

# # data['Context_length'] = data['Context'].apply(len)
# # filtered_data = data[data['Context_length'] <= 1500]
# # ln_Response = filtered_data['Response'].apply(len)
# # filtered_data = filtered_data[ln_Response <= 4000]

# filtered_data = data

# # print(torch.cuda.device_count())


# max_seq_length = 5020
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Llama-3.2-1B-bnb-4bit",
#     max_seq_length=max_seq_length,
#     load_in_4bit=True,
#     dtype=None,
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     lora_alpha=16,
#     lora_dropout=0,
#     target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
#     use_rslora=True,
#     use_gradient_checkpointing="unsloth",
#     random_state = 32,
#     loftq_config = None,
# )
# print(model.print_trainable_parameters())

# data_prompt = """Analyze the provided text from a mental health perspective. Identify any indicators of emotional distress, coping mechanisms, or psychological well-being. Highlight any potential concerns or positive aspects related to mental health, and provide a brief explanation for each observation.

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token
# def formatting_prompt(examples):
#     inputs       = examples["Context"]
#     outputs      = examples["Response"]
#     texts = []
#     for input_, output in zip(inputs, outputs):
#         text = data_prompt.format(input_, output) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }


# training_data = Dataset.from_pandas(filtered_data)
# training_data = training_data.map(formatting_prompt, batched=True)

# print(len(training_data))


# trainer=SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=training_data,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=2,
#     packing=True,
#     args=TrainingArguments(
#         learning_rate=3e-4,
#         lr_scheduler_type="linear",
#         per_device_train_batch_size=16,
#         gradient_accumulation_steps=8,
#         num_train_epochs=40,
#         fp16=not is_bfloat16_supported(),
#         bf16=is_bfloat16_supported(),
#         logging_steps=1,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         warmup_steps=10,
#         output_dir="output",
#         seed=0,
#     ),
# )

# trainer.train()


# # INFERENCE

# text="I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"
# model = FastLanguageModel.for_inference(model)
# inputs = tokenizer(
# [
#     data_prompt.format(
#         #instructions
#         text,
#         #answer
#         "",
#     )
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 5020, use_cache = True)
# answer=tokenizer.batch_decode(outputs)
# answer = answer[0].split("### Response:")[-1]
# print("Answer of the question is:", answer)

# model.save_pretrained("./finetuned_model/1B_finetuned_llama3.2")
# tokenizer.save_pretrained("./finetuned_model/1B_finetuned_llama3.2")

