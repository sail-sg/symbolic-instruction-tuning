# Perform Zero-shot Tabel Reasoning using ChatGPT / CodeX

## Install dependencies

```bash
pip install -r requirements.txt
```

## Build the evaluation file

Please remember to replace the `build_wtq_zero_dataset` function with your dersired function in `build_eval_dataset.py` before running the following command.

```bash
python build_eval_dataset.py
```

## Run the evaluation

For ChatGPT, please use the following command

```bash
python infer_chatgpt.py
```

For CodeX, please use the following command

```bash
python infer_codex.py
```

## Evaluate the prediction

```bash
python evaluate.py
```