import json
from eval_utils import get_denotation_accuracy_binder, get_denotation_accuracy


def normalize_output(output):
    prefix_list = ["The answer is", "the answer is", "Answer: ", "Extracted Answer: ", "Answer :", ": "]
    for prefix in prefix_list:
        if output.startswith(prefix):
            output = output[len(prefix):]
    output = output.strip(".").strip()
    output = output.replace("; ", ", ")
    return output


def evaluate_file(file_path):
    predictions, ground_truth, questions = [], [], []
    with open(file_path, "r", encoding="utf8") as f:
        json_lines = f.readlines()
        for idx, line in enumerate(json_lines):
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj["prediction"], list):
                    json_obj["prediction"] = json_obj["prediction"][0]
                predictions.append(normalize_output(json_obj["prediction"]))
                ground_truth.append(normalize_output(json_obj["ground_truth"]))
                if "question" in json_obj:
                    questions.append(json_obj["question"])
                else:
                    questions.append(json_obj["metadata"]["question"])
            except Exception as e:
                print(e)
    acc = get_denotation_accuracy_binder(predictions, ground_truth, questions)
    print("Total examples: {}".format(len(predictions)))
    print("Denotation Accuracy: {}".format(acc))


if __name__ == '__main__':
    # if the prediction file is generated by codex, it should have the suffix ".json.codex', otherwise ".json.chatgpt"
    evaluate_file("<<PREDICTION_FILE_PATH>>")