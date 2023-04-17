import re
from typing import Dict, List
from preprocessor.table_utils.table_linearzie import *
from preprocessor.table_utils.table_truncate import *
from constant import DEFAULT_TEMPLATE, DEL
from jinja2 import Template


class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = DEL):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter

    def process_input(self, table: Dict, question: str, template: str=None,
                      **kwargs) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        if "{table}" in template:
            raise ValueError("You should not use {table} in template since you are using Jinja2 template rendering")

        if template is None:
            template = DEFAULT_TEMPLATE
            print("You do not specify the template, so we use the default template: {}".format(template))

        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            # use template to truncate table, especially for few-shot examples
            truncate_func.truncate_table(table, question + template, [])
        # linearize a table into a string
        linear_table = self.table_linearize_func.process_table(table)

        # use Jinja2 to render the template
        template = Template(template)
        joint_input = template.render(table=linear_table,
                                      question=question,
                                      **kwargs)
        return joint_input

    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        output = self.target_delimiter.join(answer)
        if output.strip() == "":
            return "@NULL@"
        else:
            return output


def get_default_processor(max_cell_length, max_input_length, model_name):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor


def get_natural_processor(max_cell_length, max_input_length, model_name):
    table_linearize_func = NaturalTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor


def get_codex_processor(max_cell_length, max_input_length, model_name):
    table_linearize_func = CodexTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor
