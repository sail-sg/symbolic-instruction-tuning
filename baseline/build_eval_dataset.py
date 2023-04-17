import json
import os
from copy import deepcopy
from typing import List
from constant import MAX_LENGTH
from datasets import load_dataset

from preprocessor.dataset_utils.wikisql_utils import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER
from preprocessor.table_utils.table_process import get_codex_processor

TEMPLATE_ROOT_DIR = "prompt_templates"


def build_wtq_zero_dataset(folder, template_files):
    os.makedirs(folder, exist_ok=True)
    # make the demonstration of the table processor as the same maximum input length for the model
    table_processor = get_codex_processor(max_cell_length=10,
                                          max_input_length=MAX_LENGTH,
                                          # using GPT2 to estimate the tokens for GPT3
                                          model_name="gpt2")
    eval_dataset = load_dataset("wikitablequestions", split="test")
    prompt_files = template_files
    for prompt_file in prompt_files:
        # the prompt_file should end with chatgpt or codex
        prompt_mode = prompt_file.split(".")[-1]
        prompt_string = open(os.path.join(TEMPLATE_ROOT_DIR, prompt_file), "r", encoding="utf8").read().strip()
        write_f = open(f"{folder}/wtq_{prompt_mode}.json", "w",
                       encoding="utf8")

        for idx, (question, table, answer) in enumerate(
                zip(eval_dataset["question"], eval_dataset["table"], eval_dataset["answers"])):
            # truncate table content
            for truncate_func in table_processor.table_truncate_funcs:
                # use template to truncate table, especially for few-shot examples
                truncate_func.truncate_table(table, question, [])
            # linearize a table into a string
            linear_table = table_processor.table_linearize_func.process_table(table)
            # get the formatted input
            model_input = prompt_string.format(question=question, table=linear_table)
            ground_truth = table_processor.process_output(answer=answer)
            write_f.write(json.dumps({"input": model_input,
                                      "output": ground_truth,
                                      "metadata": {
                                          "question": question,
                                          "table": table,
                                          "answer": answer
                                      }}) + "\n")
        write_f.close()


def build_wikisql_zero_dataset(folder, template_files):
    os.makedirs(folder, exist_ok=True)
    table_processor = get_codex_processor(max_cell_length=10,
                                          max_input_length=MAX_LENGTH,
                                          # using GPT2 to estimate the tokens for GPT3
                                          model_name="gpt2")

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

    eval_dataset = load_dataset("wikisql", split="test")
    prompt_files = template_files

    for prompt_file in prompt_files:
        # the prompt_file should end with chatgpt or codex
        prompt_mode = prompt_file.split(".")[-1]
        prompt_string = open(os.path.join(TEMPLATE_ROOT_DIR, prompt_file), "r", encoding="utf8").read().strip()
        write_f = open(f"{folder}/wikisql_{prompt_mode}.json", "w",
                       encoding="utf8")

        for question, table, example_sql in zip(eval_dataset["question"], eval_dataset["table"], eval_dataset["sql"]):
            tapas_table = _convert_table_types(table)
            answer: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)
            # truncate table content
            for truncate_func in table_processor.table_truncate_funcs:
                # use template to truncate table, especially for few-shot examples
                truncate_func.truncate_table(table, question, [])
            # linearize a table into a string
            linear_table = table_processor.table_linearize_func.process_table(table)
            # get the formatted input
            model_input = prompt_string.format(question=question.strip("?"), table=linear_table)
            ground_truth = table_processor.process_output(answer=answer)
            write_f.write(json.dumps({"input": model_input,
                                      "output": ground_truth,
                                      "metadata": {
                                          "question": question,
                                          "table": table,
                                          "answer": answer
                                      }}) + "\n")
        write_f.close()


def build_tabfact_zero_dataset(folder, template_files):
    os.makedirs(folder, exist_ok=True)
    table_processor = get_codex_processor(max_cell_length=10,
                                          max_input_length=MAX_LENGTH,
                                          # using GPT2 to estimate the tokens for GPT3
                                          model_name="gpt2")
    eval_dataset = load_dataset("tab_fact", "tab_fact", split="test")

    def table2dict(table_str):
        table_list = [line.split('#') for line in table_str.strip().split('\n')]
        table_dict = {'header': table_list[0], 'rows': table_list[1:]}
        return table_dict

    tab_fact_mapping = {0: 'no', 1: 'yes'}

    prompt_files = template_files

    for prompt_file in prompt_files:
        # the prompt_file should end with chatgpt or codex
        prompt_mode = prompt_file.split(".")[-1]
        prompt_string = open(os.path.join(TEMPLATE_ROOT_DIR, prompt_file), "r", encoding="utf8").read().strip()
        write_f = open(f"{folder}/tabfact_{prompt_mode}.json", "w",
                       encoding="utf8")

        for idx, (question, table_text, label, caption) in enumerate(
                zip(eval_dataset["statement"], eval_dataset["table_text"], eval_dataset["label"],
                    eval_dataset["table_caption"])):
            # get table dict from table text
            table = table2dict(table_text)
            for truncate_func in table_processor.table_truncate_funcs:
                # use template to truncate table, especially for few-shot examples
                truncate_func.truncate_table(table, question, [])
            # linearize a table into a string
            linear_table = table_processor.table_linearize_func.process_table(table)
            # get the formatted input
            template_input = prompt_string.format(question=question.strip("?"), table=linear_table, caption=caption)
            template_output = tab_fact_mapping[label]
            write_f.write(json.dumps({"input": template_input,
                                      "output": template_output,
                                      "metadata": {
                                          "question": question,
                                          "table": table,
                                          "label": label
                                      }}) + "\n")
        write_f.close()


def build_sqa_zero_dataset(folder, template_files):
    os.makedirs(folder, exist_ok=True)
    # make the demonstration of the table processor as the same maximum input length for the model
    table_processor = get_codex_processor(max_cell_length=10,
                                          max_input_length=MAX_LENGTH,
                                          # using GPT2 to estimate the tokens for GPT3
                                          model_name="gpt2")
    eval_dataset = load_dataset("msr_sqa", split="test")
    prompt_files = template_files
    for prompt_file in prompt_files:
        # the prompt_file should end with chatgpt or codex
        prompt_mode = prompt_file.split(".")[-1]
        prompt_string = open(os.path.join(TEMPLATE_ROOT_DIR, prompt_file), "r", encoding="utf8").read().strip()
        write_f = open(f"{folder}/sqa_{prompt_mode}.json", "w",
                       encoding="utf8")

        for idx, (history, table_header, table_values, answer) in enumerate(
                zip(eval_dataset["question_and_history"], eval_dataset["table_header"],
                    eval_dataset["table_data"], eval_dataset["answer_text"])):
            table = {"header": table_header,
                     "rows": table_values}
            question = " ".join(history)
            # truncate table content
            for truncate_func in table_processor.table_truncate_funcs:
                # use template to truncate table, especially for few-shot examples
                truncate_func.truncate_table(table, question, [])
            # linearize a table into a string
            linear_table = table_processor.table_linearize_func.process_table(table)
            # get the formatted input
            model_input = prompt_string.format(question=question, table=linear_table)
            ground_truth = table_processor.process_output(answer=answer)
            write_f.write(json.dumps({"input": model_input,
                                      "output": ground_truth,
                                      "metadata": {
                                          "question": question,
                                          "table": table,
                                          "answer": answer
                                      }}) + "\n")
        write_f.close()


if __name__ == '__main__':
    build_wtq_zero_dataset("eval_dataset", template_files=["wtq.chatgpt",
                                                           "wtq.codex"])
