import json
import os
import shutil

from datasets import load_dataset
import re
from typing import Dict, List
import sys
from preprocessor.table_utils.table_process import get_default_processor
from preprocessor.dataset_utils.wikisql_utils import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER
from eval_scripts.prompt_collection import *
from preprocessor import *
from bbh_constants import *
from tqdm import tqdm
from copy import deepcopy
import hashlib
from jinja2 import Template
import requests
import tarfile
from transformers import AutoTokenizer
from constant import MAX_LENGTH
import pandas as pd


def build_tabfact_zero_dataset(dataset_name, folder):
    prompt_templates = get_tabfact_prompt_templates()
    os.makedirs(f"{folder}/{dataset_name}", exist_ok=True)
    table_processor = get_default_processor(max_cell_length=10,
                                            max_input_length=MAX_LENGTH,
                                            model_name="google/flan-t5-xl")

    def table2dict(table_str):
        table_list = [line.split('#') for line in table_str.strip().split('\n')]
        table_dict = {'header': table_list[0], 'rows': table_list[1:]}
        return table_dict

    tab_fact_mapping = {0: 'no', 1: 'yes'}
    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        for split_name in ["validation", "test"]:
            dataset = load_dataset("tab_fact", "tab_fact")
            eval_dataset = dataset[split_name]
            write_file = f"{folder}/{dataset_name}/{dataset_name}_{split_name}_zero_template_{idx}.json"
            with open(write_file, "w") as write_f:
                for _, (question, table_text, label) in enumerate(
                        zip(eval_dataset["statement"], eval_dataset["table_text"], eval_dataset["label"])):
                    # get table dict from table text
                    table = table2dict(table_text)
                    template_input = table_processor.process_input(table=table,
                                                                   question=question,
                                                                   template=prompt_template)
                    template_output = tab_fact_mapping[label]
                    write_f.write(json.dumps({"input": template_input, "output": template_output}) + "\n")
        write_f.close()


def build_sqa_zero_dataset(dataset_name, folder):
    prompt_templates = get_sqa_prompt_templates()
    os.makedirs(f"{folder}/{dataset_name}", exist_ok=True)
    table_processor = get_default_processor(max_cell_length=10,
                                            max_input_length=MAX_LENGTH,
                                            model_name="google/flan-t5-xl")
    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        dataset = load_dataset("msr_sqa")
        for split_name in ["validation", "test"]:
            eval_dataset = dataset[split_name]
            write_file = f"{folder}/{dataset_name}/{dataset_name}_{split_name}_zero_template_{idx}.json"
            with open(write_file, "w") as write_f:
                for _, (history, table_header, table_values, answer) in enumerate(
                        zip(eval_dataset["question_and_history"], eval_dataset["table_header"],
                            eval_dataset["table_data"], eval_dataset["answer_text"])):
                    template_input = table_processor.process_input(table={"header": table_header,
                                                                          "rows": table_values
                                                                          }, question=" ".join(history),
                                                                   template=prompt_template)
                    template_output = table_processor.process_output(answer=answer)
                    write_f.write(json.dumps({"input": template_input, "output": template_output}) + "\n")


def build_wtq_zero_dataset(dataset_name, folder):
    prompt_templates = get_wtq_prompt_templates()
    os.makedirs(f"{folder}/{dataset_name}", exist_ok=True)
    table_processor = get_default_processor(max_cell_length=10,
                                            max_input_length=MAX_LENGTH,
                                            model_name="google/flan-t5-xl")

    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        for split_name in ["validation", "test"]:
            write_file = f"{folder}/{dataset_name}/{dataset_name}_{split_name}_zero_template_{idx}.json"
            with open(write_file, "w") as write_f:
                dataset = load_dataset("wikitablequestions")
                eval_dataset = dataset[split_name]
                for _, (question, table, answer) in enumerate(
                        zip(eval_dataset["question"], eval_dataset["table"], eval_dataset["answers"])):
                    template_input = table_processor.process_input(table=table, question=question,
                                                                   template=prompt_template)
                    template_output = table_processor.process_output(answer=answer)
                    write_f.write(json.dumps({"input": template_input, "output": template_output}) + "\n")
                write_f.close()


def build_wikisql_zero_dataset(dataset_name, folder):
    prompt_templates = get_wikisql_prompt_templates()
    os.makedirs(f"{folder}/{dataset_name}", exist_ok=True)
    table_processor = get_default_processor(max_cell_length=10,
                                            max_input_length=MAX_LENGTH,
                                            model_name="google/flan-t5-xl")

    # this function is specific for WikiSQL since the util function need the data structure
    # to retrieve the WikiSQL answer for each question
    def _convert_table_types(_table):
        """Runs the type converter over the table cells."""
        ret_table = deepcopy(_table)
        types = ret_table["types"]
        ret_table["real_rows"] = ret_table["rows"]
        typed_rows = []
        for row in ret_table["rows"]:
            typed_row = []
            for column, cell_value in enumerate(row):
                typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
            typed_rows.append(typed_row)
        ret_table["rows"] = typed_rows
        return ret_table

    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        for split_name in ["validation", "test"]:
            dataset = load_dataset("wikisql")
            eval_dataset = dataset[split_name]
            write_path = f"{folder}/{dataset_name}/{dataset_name}_{split_name}_zero_template_{idx}.json"
            with open(write_path, "w") as write_f:
                for question, table, example_sql in zip(eval_dataset["question"], eval_dataset["table"],
                                                        eval_dataset["sql"]):
                    example_input = table_processor.process_input(table=table,
                                                                  question=question,
                                                                  template=prompt_template)
                    tapas_table = _convert_table_types(table)
                    answer_list: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)
                    # you can choose other delimiters to split each answer
                    example_output = table_processor.process_output(answer_list)
                    write_f.write(json.dumps({"input": example_input, "output": example_output}) + "\n")


def build_svamp_zero_dataset(dataset_name, folder):
    prompt_templates = get_svamp_prompt_templates()
    os.makedirs(f"{folder}/svamp", exist_ok=True)
    url_link = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/dev.csv"
    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        template = Template(prompt_template)
        write_file = f"{folder}/svamp/{dataset_name}_zero_template_{idx}.json"
        # read the csv file from full_link
        dataset_df = pd.read_csv(url_link)
        with open(write_file, "w") as write_f:
            for _, row in dataset_df.iterrows():
                question = row["Question"]
                numbers = row["Numbers"].split(" ")
                for i in range(len(numbers)):
                    # convert string to int
                    numbers[i] = str(int(float((numbers[i]))))
                    question = question.replace("number" + str(i), numbers[i])
                template_input = template.render(question=question)
                template_output = str(int(row["Answer"]))
                write_f.write(json.dumps({
                    "input": template_input,
                    "output": template_output}) + "\n")


def build_bbh_fewshot_dataset(dataset_name, folder):
    assert dataset_name in ALL_BBH_TEMPLATES
    prompt_templates = ALL_BBH_TEMPLATES[dataset_name]

    for idx, prompt_template in enumerate(prompt_templates):
        print("Current prompt template: ", prompt_template)
        template = Template(prompt_template)
        data_file = f"data_generator/bbh/{dataset_name}.json"

        with open(data_file, "r") as fs:
            dataset = json.load(fs)
            examples = dataset['examples']
            # "input" "target"
            os.makedirs(f"{folder}/bbh/", exist_ok=True)
            os.makedirs(f"{folder}/bbh/{dataset_name}", exist_ok=True)
            write_f = open(f"{folder}/bbh/{dataset_name}/{dataset_name}_zero_template_{idx}.json", 'w')
            for idx, example in enumerate(examples):
                inputs = example["input"]
                sample_output = example["target"]
                template_input = template.render(inputs=inputs)
                template_output = sample_output
                write_f.write(json.dumps({
                    "input": template_input,
                    "output": template_output}) + "\n")
            write_f.close()


def build_mmlu_fewshot_dataset(dataset_name, folder):
    from eval_scripts.mmlu_constants import subcategories, categories

    # using tokenizer to tokenize the input
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    # download the mmlu dataset
    download_link = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    os.makedirs(f"{folder}/{dataset_name}", exist_ok=True)
    if not os.path.exists(f"{folder}/{dataset_name}/mmlu_few.tar"):
        # use requests to download the data
        r = requests.get(download_link, allow_redirects=True)
        open(f"{folder}/{dataset_name}/mmlu_few.tar", 'wb').write(r.content)
    # extract the tar file
    tar = tarfile.open(f"{folder}/{dataset_name}/mmlu_few.tar", "r:")
    tar.extractall(f"{folder}/{dataset_name}")
    tar.close()
    # read the csv file
    data_folder = f"{folder}/{dataset_name}/data"

    choices = ["A", "B", "C", "D"]

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += format_example(train_df, i)
        return prompt

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_folder, "test"))
            if "_test.csv" in f
        ]
    )

    few_shot_example = 5
    for subject in subjects:
        print("Current subject: ", subject)
        subject_folder = os.path.join(f"{folder}/{dataset_name}/{dataset_name}_{subject}")
        os.makedirs(subject_folder, exist_ok=True)
        # 5-shot dataset
        dev_df = pd.read_csv(
            os.path.join(data_folder, "dev", subject + "_dev.csv"), header=None
        )[: few_shot_example]
        test_df = pd.read_csv(
            os.path.join(data_folder, "test", subject + "_test.csv"), header=None
        )
        cors = []

        with open(os.path.join(subject_folder, f"{subject}_few_template_{0}.json"), "w") as f:
            for i in tqdm(range(test_df.shape[0])):
                # get prompt and make sure it fits
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, few_shot_example)
                prompt = train_prompt + prompt_end

                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                all_cors.append(cors)

                # try to load as many as examples
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                k = few_shot_example
                while input_ids.shape[-1] > MAX_LENGTH:
                    k -= 1
                    train_prompt = gen_prompt(dev_df, subject, k)
                    prompt = train_prompt + prompt_end
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

                label = test_df.iloc[i, test_df.shape[1] - 1]
                f.write(json.dumps({"input": prompt, "output": label}) + "\n")


if __name__ == '__main__':
    build_wtq_zero_dataset(dataset_name="wtq", folder="eval")
