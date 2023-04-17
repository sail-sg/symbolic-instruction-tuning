import json
import os
from functools import partial
from logging import getLogger

import torch
from datasets import disable_caching
from datasets import load_dataset, DownloadMode
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorWithPadding
from eval_utils import *
from constant import *
from preprocessor.default import preprocess_function as default_preprocess_function
from accelerate import Accelerator

logger = getLogger(__name__)


def main(model_name, batch_size, data_file, eval_func_name):
    torch.cuda.empty_cache()
    accelerator = Accelerator()
    # by default as test
    split_name = "test"
    dataset = load_dataset("json", data_files={split_name: data_file})

    print("Evaluating {} on file {} \n\n".format(model_name, data_file))
    # print the demo case of dataset
    if accelerator.is_main_process:
        for i in range(3):
            print("=================== Here are some samples =====================")
            print(dataset[split_name]["input"][i])
            print(dataset[split_name]["output"][i])
            print("=================== Stop =====================")
            print()

    device = accelerator.device
    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl")
    if "xxl" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    eval_dataset = dataset[split_name]
    preprocess_func = partial(default_preprocess_function,
                              tokenizer=tokenizer,
                              lowercase=False)
    # eval_dataset = eval_dataset.select(range(10))
    with accelerator.main_process_first():
        eval_dataset_processed = eval_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=4,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False
        )
    data_loader = DataLoader(
        eval_dataset_processed,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=4,
        pin_memory=True
    )
    # prepare the model and data_loader with accelerator
    model, data_loader = accelerator.prepare(model, data_loader)
    outputs = []
    ground_truths = []

    all_model_predictions, all_targets, all_inputs = [], [], []
    for step, inputs in tqdm(enumerate(data_loader)):
        model_inputs, model_targets = inputs["input_ids"].to(device), inputs["labels"].to(device)
        model_predictions = accelerator.unwrap_model(model).generate(
            input_ids=model_inputs,
            max_length=256
        )
        # padding the prediction to the same length as the target
        model_predictions = accelerator.pad_across_processes(model_predictions, dim=1, pad_index=tokenizer.pad_token_id)
        # Gather all predictions and labels across all processes
        distri_predictions, distri_inputs, distri_targets = accelerator.gather_for_metrics((model_predictions,
                                                                                            model_inputs,
                                                                                            model_targets))
        all_model_predictions.append(distri_predictions.cpu().numpy())
        all_inputs.append(distri_inputs.cpu().numpy())
        all_targets.append(distri_targets.cpu().numpy())

    accelerator.wait_for_everyone()
    # create the output
    os.makedirs("outputs", exist_ok=True)
    if accelerator.is_main_process:
        log_name = model_name.split("/")[-1]
        log_dataset_name = data_file.split("/")[-1] if data_file.endswith(".json") else data_file

        result_f = open("outputs/{}_{}.json".format(log_dataset_name,
                                                    log_name),
                        "w", encoding="utf8")

        for idx, predict_seq in tqdm(enumerate(all_model_predictions)):
            # Decode text
            output = tokenizer.batch_decode(predict_seq, skip_special_tokens=True)
            # Remove all text after the stop token
            real_input = tokenizer.batch_decode(all_inputs[idx], skip_special_tokens=True)
            ground_truth = tokenizer.batch_decode(all_targets[idx], skip_special_tokens=True)
            for input_example, output_example, ground_truth_example in zip(real_input, output, ground_truth):
                result_f.write(json.dumps({"input": input_example,
                                           "prediction": output_example,
                                           "ground_truth": ground_truth_example}) + "\n")
            outputs.extend(output)
            ground_truths.extend(ground_truth)

        eval_func = globals()[eval_func_name]
        perf = eval_func(outputs, ground_truths)
        log_out = "Evaluating {} on {}, Eval func is {}, Performance is: {}".format(
            model_name,
            data_file,
            eval_func_name,
            perf)
        print(log_out)
        result_f.close()

    # we should wait for the main process to upload the log, and then to the next inference file
    accelerator.wait_for_everyone()


def main_wrapper():
    checkpoint_path = "sail/tapex-zero-large"
    eval_file = "https://huggingface.co/datasets/sail/symbolic-instruction-tuning/blob/main/test/wtq_tapex_large.json"
    main(model_name=checkpoint_path,
         data_file=eval_file,
         batch_size=8,
         eval_func_name="get_denotation_accuracy")


if __name__ == '__main__':
    main_wrapper()
