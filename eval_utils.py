import re
import unicodedata
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import isnan, isinf
from typing import List

import evaluate
import recognizers_suite
from recognizers_suite import Culture

culture = Culture.English

delimiter = ","


# define example evaluation
def evaluate_example(predict_str: str, ground_str: str):
    ground_str = ground_str.lower()
    predict_str = predict_str.lower()
    predict_spans = predict_str.split(delimiter)
    ground_spans = ground_str.split(delimiter)
    predict_values = defaultdict(lambda: 0)
    ground_values = defaultdict(lambda: 0)
    for span in predict_spans:
        try:
            predict_values[float(span)] += 1
        except ValueError:
            predict_values[span.strip()] += 1
    for span in ground_spans:
        try:
            ground_values[float(span)] += 1
        except ValueError:
            ground_values[span.strip()] += 1
    _is_correct = predict_values == ground_values
    return _is_correct


def get_denotation_accuracy(predictions: List[str], references: List[str], **kwargs):
    assert len(predictions) == len(references)
    correct_num = 0
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str, ground_str)
        if is_correct:
            correct_num += 1
    return correct_num / len(predictions)


def get_denotation_accuracy_binder(predictions: List[str], references: List[str], questions: List[str]):
    assert len(predictions) == len(references)
    correct_num = 0
    for predict_str, ground_str, question in zip(predictions, references, questions):
        is_correct = evaluate_example_binder(predict_str, ground_str, question=question)
        if is_correct:
            correct_num += 1
    return correct_num / len(predictions)


def get_exact_match(predictions: List[str], references: List[str], **kwargs):
    """
    Exact match as the default evaluation
    """
    assert len(predictions) == len(references)
    correct_num = 0
    for prediction, reference in zip(predictions, references):
        if prediction.lower() == reference.lower():
            correct_num += 1
    return correct_num / len(predictions)


def get_exact_match_option(predictions: List[str], references: List[str], **kwargs):
    """
    Exact match as the default evaluation
    """
    assert len(predictions) == len(references)
    correct_num = 0
    for prediction, reference in zip(predictions, references):
        if prediction.lower().replace("(", '').replace(")", '') == reference.lower().replace("(", '').replace(")", ''):
            correct_num += 1
    return correct_num / len(predictions)


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


################ String Normalization ################

def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.
        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.
        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.
        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.
    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values
    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end:idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = ["datetime",
                             "number",
                             # "ordinal",
                             # "percentage",
                             # "age",
                             # "currency",
                             # "dimension",
                             # "temperature",
                             ]

    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(recognizers_suite, "recognize_{}".format(recognition_type))(user_input,
                                                                                              culture)  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex'])  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[:-len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input


def evaluate_example_official(predict_list: List, ground_truth: List):
    # for pred, gt in zip(predict_list, ground_truth):
    predict_spans = [str(val).lower() for val in predict_list]
    predict_spans = to_value_list(predict_spans)
    ground_spans = to_value_list(ground_truth)
    ret = check_denotation(target_values=ground_spans, predicted_values=predict_spans)
    return ret


def evaluate_example_binder(predict_list: str, ground_truth: str, allow_semantic=True, question: str = None):
    pred = [str(p).lower().strip() for p in predict_list.split(delimiter)]
    gold = [str(g).lower().strip() for g in ground_truth.split(delimiter)]

    if not allow_semantic:
        # WikiTQ eval w. string normalization using recognizer
        pred = [str_normalize(span) for span in pred]
        gold = [str_normalize(span) for span in gold]
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)
    else:
        assert isinstance(question, str)
        question = re.sub('\s+', ' ', question).strip().lower()
        pred = [str_normalize(span) for span in pred]
        gold = [str_normalize(span) for span in gold]
        pred = sorted(list(set(pred)))
        gold = sorted(list(set(gold)))
        # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
        if len(pred) == 1 and len(gold) == 1:
            if (pred[0] == '0' and gold[0] == 'no') \
                    or (pred[0] == '1' and gold[0] == 'yes'):
                return True
            question_tokens = question.split()
            try:
                pos_or = question_tokens.index('or')
                token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                if (pred[0] == '0' and gold[0] == token_after_or) \
                        or (pred[0] == '1' and gold[0] == token_before_or):
                    return True
            except Exception as e:
                pass
        # (2) Number value (allow units) and Date substring match
        if len(pred) == 1 and len(gold) == 1:
            NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
            DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
            DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
            p, g = pred[0], gold[0]
            # Restore `duration` type, e.g., from 'P3Y' -> '3'
            if re.match(DURATION_PATTERN, p):
                p = re.match(DURATION_PATTERN, p).group(2)
            if re.match(DURATION_PATTERN, g):
                g = re.match(DURATION_PATTERN, g).group(2)
            match = False
            num_flag, date_flag = False, False
            # Number w. unit match after string normalization.
            # Either pred or gold being number w. units suffices it.
            if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                num_flag = True
            # Date match after string normalization.
            # Either pred or gold being date suffices it.
            if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                date_flag = True
            if num_flag:
                p_set, g_set = set(p.split()), set(g.split())
                if p_set.issubset(g_set) or g_set.issubset(p_set):
                    match = True
            if date_flag:
                p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                if p_set.issubset(g_set) or g_set.issubset(p_set):
                    match = True
            if match:
                return True
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)


def get_bleu_4(predictions: List[str], references: List[str], max_order=4, smooth=False, **kwargs):
    bleu = evaluate.load("bleu")

    predictions_group = []
    references_group = []
    cur_pred = None
    cur_refers = []
    for prediction, reference in zip(predictions, references):
        if cur_pred is None:
            cur_pred = prediction
            cur_refers.append(reference)
        elif cur_pred != prediction:
            predictions_group.append(cur_pred)
            references_group.append(cur_refers)
            # update new group
            cur_pred = prediction
            cur_refers = [reference]
        else:
            cur_refers.append(reference)
    if cur_pred is not None:
        predictions_group.append(cur_pred)
        references_group.append(cur_refers)
    assert len(predictions_group) == len(references_group)
    results = bleu.compute(predictions=predictions_group, references=references_group)
    bleu_score = results['bleu']
    return bleu_score
