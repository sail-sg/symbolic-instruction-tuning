import json
import os
import time

import openai
from tqdm import tqdm

from constant import OPENAI_API_KEY


def run_codex_api(model_input):
    result = None
    while result is None:
        try:
            result = openai.Completion.create(
                engine="code-davinci-002",
                prompt=model_input,
                api_key=OPENAI_API_KEY,
                temperature=0.0,
                max_tokens=128,
                n=1,
                stop=["\n\n", "\n"]
            )
        except Exception as e:
            print(e, 'Retry.')
            time.sleep(5)
    model_output = result["choices"][0]["text"]
    return model_output


def run_codex_prediction(test_file):
    print(f"Running codex on {test_file} ...")
    output_file = test_file.replace(".json", ".json.codex")
    print(f"Output file: {output_file} ...")
    if os.path.exists(output_file):
        # test how many examples have been processed
        passed_cases = open(output_file, "r").readlines()
        if not passed_cases[-1].endswith("\n"):
            # this line is incomplete, remove it in the file
            passed_cases = passed_cases[:-1]
            open(output_file, "w").writelines(passed_cases)
        start_idx = len(passed_cases)
    else:
        start_idx = 0
    print(f"Start from {start_idx} ...")
    # always using append mode
    with open(test_file, "r") as f, open(output_file, "a") as output_f:
        for idx, line in tqdm(enumerate(f.readlines()[start_idx:])):
            data = json.loads(line)
            model_input = data["input"]
            metadata = data["metadata"]
            model_output = run_codex_api(model_input)
            output_f.write(json.dumps({
                "prediction": model_output,
                "ground_truth": data["output"].strip(),
                "input": model_input,
                "metadata": metadata
            }) + "\n")
            if idx % 100 == 0:
                print(model_output)


if __name__ == '__main__':
    run_codex_prediction("<<TEST_FILE_PATH>>")
