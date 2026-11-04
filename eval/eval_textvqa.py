"""VQA-style evaluation for TextVQA, and export of the correctly answered samples.

The exported samples are what the information-flow analysis runs on, so what counts as
"correct" here decides which data enters the paper.
"""

import argparse
import json
import re

import contractions
from word2number import w2n

# How many of the ten annotator answers a prediction must match to count as correct.
# The standard VQA score is min(matches / 3, 1); the original code kept every sample with
# a score above zero, i.e. a single annotator out of ten. Raise this to tighten the set.
MIN_MATCHES = 1

# The original code tested  gt in pred,  so a long answer counted as correct merely by
# containing the gold string somewhere. On the shipped data only 7 of 2680 samples relied
# on that, so exact match is the default. Set to True to reproduce the looser behaviour.
ALLOW_SUBSTRING_MATCH = False

_NUMBER_WORD = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|trillion"
)
_NUMBER_RE = re.compile(
    rf"\b(?:{_NUMBER_WORD})(?:[\s-](?:{_NUMBER_WORD}))*\b"
)


def _convert_number_words(match):
    try:
        return str(w2n.word_to_num(match.group(0)))
    except ValueError:
        return match.group(0)


def preprocess_text(text):
    """Normalise an answer the way the VQA protocol does."""
    text = text.lower()
    text = contractions.fix(text)
    text = _NUMBER_RE.sub(_convert_number_words, text)

    # drop articles
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # drop periods unless they sit inside a decimal number
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # punctuation becomes a space, except apostrophes, colons, and commas inside numbers.
    # The bounds matter: a leading comma made the old code read the LAST character of the
    # string, and a trailing comma ran off the end with an IndexError.
    def replace_punctuation(m):
        if m.group(0) == ",":
            before, after = m.start() - 1, m.end()
            if before >= 0 and after < len(text) and text[before].isdigit() and text[after].isdigit():
                return ","
            return ""
        return " "

    text = re.sub(r"[^\w\s':,]", replace_punctuation, text)
    return re.sub(r"\s+", " ", text).strip()


def compute_matches(pred_answer, gt_answers):
    """Number of annotator answers the prediction agrees with."""
    pred = preprocess_text(pred_answer)
    golds = [preprocess_text(gt) for gt in gt_answers]

    matches = 0
    for gold in golds:
        # An answer such as "the" normalises to the empty string, and `"" in pred` is
        # always True: those used to match every prediction. Skip them.
        if not gold:
            continue
        if pred == gold or (ALLOW_SUBSTRING_MATCH and gold in pred):
            matches += 1
    return matches


def vqa_accuracy(matches):
    return min(1.0, matches / 3.0)


def evaluate_textvqa(results_path, correct_out_path, wrong_out_path, verbose=False):
    with open(results_path, "r") as f:
        data = json.load(f)

    total_score = 0.0
    correct_answers = []
    wrong_answers = []

    for entry in data:
        matches = compute_matches(entry["answer"], entry["gt_answer"])
        score = vqa_accuracy(matches)
        total_score += score

        if matches >= MIN_MATCHES:
            correct_answers.append(entry)
        else:
            wrong_answers.append(entry)

        if verbose:
            mark = "ok" if matches >= MIN_MATCHES else "x "
            print(f"[{mark}] pred: {entry['answer']!r} | gt: {entry['gt_answer']} | matches: {matches}")

    num_questions = len(data)
    avg_accuracy = total_score / num_questions if num_questions else 0.0
    print(f"Evaluated {num_questions} questions.")
    print(f"Average VQA accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Kept as correct ({MIN_MATCHES}+ matching annotators): {len(correct_answers)}")

    with open(correct_out_path, "w") as f:
        json.dump(correct_answers, f, indent=2)

    with open(wrong_out_path, "w") as f:
        json.dump(wrong_answers, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TextVQA results.")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to the JSON file with TextVQA results.")
    parser.add_argument("--correct_out", type=str, required=True,
                        help="Path to save correctly answered samples.")
    parser.add_argument("--wrong_out", type=str, required=True,
                        help="Path to save wrongly answered samples.")
    parser.add_argument("--verbose", action="store_true", help="Print every sample.")
    args = parser.parse_args()
    evaluate_textvqa(args.results, args.correct_out, args.wrong_out, verbose=args.verbose)
