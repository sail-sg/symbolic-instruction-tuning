import json
import os
import signal

from revChatGPT.V3 import Chatbot
from tqdm import tqdm
from constant import OPENAI_API_KEY


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")


def run_chatgpt_api(model_input):
    # Set the signal handler and a 5-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    # one request at most 30s
    signal.alarm(30)
    chatbot = Chatbot(api_key=OPENAI_API_KEY, temperature=0.0)
    response = chatbot.ask(model_input)
    return response


def run_chatgpt_prediction(test_file):
    print("Running ChatGPT on test file: {}".format(test_file))
    output_file = test_file.replace(".json", ".json.chatgpt")
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
    # always using append mode
    output_f = open(output_file, "a")
    predictions, ground_truths = [], []
    print("Start from {}".format(start_idx))
    with open(test_file, "r") as f:
        for idx, line in tqdm(enumerate(f.readlines()[start_idx:])):
            data = json.loads(line)
            model_input = data["input"]
            metadata = data["metadata"]
            model_output = None
            while model_output is None:
                try:
                    model_output = run_chatgpt_api(model_input)
                except Exception as e:
                    print(e)
                finally:
                    signal.alarm(0)

            predictions.append(model_output.strip())
            ground_truths.append(data["output"].strip())
            if idx % 10 == 0:
                print(model_output)
            output_f.write(json.dumps({
                "prediction": model_output.strip(),
                "ground_truth": data["output"].strip(),
                "input": model_input,
                "metadata": metadata
            }) + "\n")
            output_f.flush()
    output_f.close()


if __name__ == '__main__':
    run_chatgpt_prediction("<<TEST_FILE_PATH>>")
