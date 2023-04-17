from jinja2 import Template

from constant import MAX_LENGTH


def preprocess_function_with_template(examples, tokenizer, template, lowercase, **kwargs):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """
    assert "input_fields" in examples
    input_fields = examples["input_fields"][0]
    inputs = []

    for idx in range(len(examples["input_fields"])):
        render_dict = {field: examples[field][idx] for field in input_fields}
        jinja_template = Template(template)
        render_input = jinja_template.render(**render_dict)
        assert "{{ " not in render_input, "Template not rendered properly!"
        inputs.append(render_input)

    if lowercase:
        inputs = [example.lower() for example in inputs]

    model_inputs = tokenizer(
        text=inputs,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        text=examples["output"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_function(examples, tokenizer, lowercase, **kwargs):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """
    if lowercase:
        examples["input"] = [example.lower() for example in examples["input"]]

    model_inputs = tokenizer(
        text=examples["input"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        text=examples["output"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
