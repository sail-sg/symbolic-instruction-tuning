

def get_svamp_prompt_templates():
    return [
        "Question: {{ question }}?\nAnswer:",
        "{{ question }}?",
        "Answer the following question:\n\n{{ question }}",
        "Answer this question:\n\n{{ question }}?",
        "Please answer this question: {{ question }}",
        "Answer the question...{{ question }}?",
        "What is the answer to this question? {{ question }}\n\n{{ answer }}",
        "Can you tell me the answer to {{ question }}?",
        "Q: {{ question }} A:",
        "Solve this math problem\n\n{{ question }}",
        "What is the solution?\n\n{{ question }}",
        "Math Problem\n{{ question }}",
        "Write down the solution for this math problem: {{ question }}"
        "What is the solution to this math problem?\n{{ question }}",
        "Math problem: {{ question }}\nWhat is the solution?",
        "{{ question }}\nSolve this problem.",
        "Problem: {{ question }}\nAnd the answer is",
        "{{ question }}. What is the answer?"
    ]


def get_sqa_prompt_templates():
    return [
        "[ {{ table }} ] Answer the following dialogue based on the above table: {{ question }}",
        "[ {{ table }} ] Answer this question based on the above table: {{ question }}",
        "<table> {{ table }} </table> Answer the following question based on the above table: {{ question }}",
        "Please answer a question about the following table \n\n{{ table }}\n\n{{ question }}",
        "Read this and answer the question\n\n{{ table }}\n\n{{ question }}",
        "[ {{ table }} ] Read the dialog and predict the answer of the last question: {{ question }}",
        "[ {{ table }} ] After reading the above table, can you answer the following questions: {{ question }}",
        "[ {{ table }} ] {{ question }}",
        "Answer the question below using the data in {{ table }}: {{ question }}",
        "Using the information in {{ table }}, what is the answer of {{ question }}?"
    ]


def get_tabfact_prompt_templates():
    return [
        "Table:\n{{ table }}\nClaim: \"{{ question }}\" Is the claim entailed by the Table? Options: - yes - no. The answer is",
        "Examine the provided table below: {{ table }} A claim has been made: \"{{ question }}\". Based on the information in the table, can we determine if this claim is accurate? Please select one of the following options: - yes - no. The correct response is:",
        "[ {{ table }} ] Verify this claim based on the above table: {{ question }}. Options: - yes - no. The answer is",
        "Here is a table: {{ table }}. Someone has stated that {{ question }}. Does the table support this statement? Please choose either: - yes - no. The answer is:",
        "Consider the information provided in this table: {{ table }} The claim \"{{ question }}\" has been made. Does the table entail this claim? Choose your answer from the options \"yes\" or \"no\". The answer is:",
        "Observe the following table: {{ table }}\n\nThere is a claim stating \"{{ question }}.\" Can we determine if this claim is valid based on the information in the table? Please select one of these options: - yes - no. The answer is:",
        "\"{{ question }}\" is a claim being made. Considering the information in the table provided below, can we verify the correctness of this claim? {{ table }} Choose your answer from the options \"yes\" or \"no\". The answer is:",
        "Read this table and verify the following claim:\n {{ table }}\n{{ question }} You can choose from the following options: - yes - no. The answer is:",
        "Based on the data provided in {{ table }}, what is the correctness of {{ question }}? Options: * yes * no. The answer is",
        "A claim suggests that \"{{ question }}\" Refer to the table given below and determine if this claim is justified based on the information present: {{ table }} Options: - yes - no. The answer is"
    ]


def get_wikisql_prompt_templates():
    return [
        "<table> {{ table }} </table> Answer this question based on the above table: {{ question }}",
        "[Table] {{ table }} [/Table] Answer this question based on the above table: {{ question }}",
        "[ {{ table }} ] Answer this question based on the above table: {{ question }}",
        "Please answer a question about the following table \n\n{{ table }}\n\n{{ question }}",
        "Here is a table: {{ table }}. Answer the following question: {{ question }}",
        "Answer the question {{ question }} based on the following table: {{ table }}",
        "{{ table }}\n\n{{ question }}",
        "{{ table }}\nAnswer this question: {{ question }}",
        "Read this table and answer this question {{ table }}\n{{ question }}",
        "What conclusions can be drawn about {{ question }} from the information presented in {{ table }}? {{ answer }}"
    ]


def get_wtq_prompt_templates():
    return [
        "[Table] {{ table }} [/Table] Answer this question based on the above table: {{ question }}",
        "[ {{ table }} ] Answer this question based on the above table: {{ question }}",
        "Here is a table: {{ table }}. Answer the following question: {{ question }}",
        "Answer the question {{ question }} based on the following table: {{ table }}",
        "{{ table }}\nAnswer this question: {{ question }}",
        "Read this table and answer this question {{ table }}\n{{ question }}",
        "Based on the data provided in {{ table }}, what is the value of {{ question }}?",
        "Using {{ table }}, answer the following question: {{ question }}.",
        "Answer the question below using the data in {{ table }}: {{ question }}.",
        "Using the information in {{ table }}, what is the answer of {{ question }}?"
    ]
