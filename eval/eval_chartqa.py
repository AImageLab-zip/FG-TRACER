"""Relaxed-accuracy evaluation for ChartQA, and export of the correctly answered samples.

The exported samples are what the information-flow analysis runs on, so what counts as
"correct" here decides which data enters the paper.
"""

import argparse
import json
import re

# A gold answer is matched exactly, after both sides are normalised (markdown markers,
# surrounding quotes, a trailing period and a leading article are stripped).
#
# The original code also accepted  gold in pred,  which let a prediction be correct merely
# by containing the gold string: "no" matches "nothing", "normal", "not available". On the
# shipped data that rule only mattered for 38 of 435 non-numeric answers, almost all of
# them markdown leftovers that normalisation now handles properly. Set this to True to
# reproduce the original, looser behaviour.
ALLOW_SUBSTRING_MATCH = False

# Relative tolerance for numeric answers, as in the ChartQA relaxed-accuracy protocol.
NUMERIC_TOLERANCE = 0.05

ANSWER_MARKERS = (
    "**FINAL ANSWER:**",
    "FINAL ANSWER:",
    "Answer:**",
    "Answer**:",
    "Answer*:",
    "Answer:",
)


def extract_final_answer(answer_text):
    """Return the text after the last answer marker, or the last non-empty line.

    The marker is searched from the end: a chain of thought often says "Answer:" while
    reasoning and again at the very end. create_chartqa.py splits on the same marker and
    must use the same occurrence, or a sample is judged on one span and analysed on another.
    """
    best_cut = None
    for marker in ANSWER_MARKERS:
        idx = answer_text.rfind(marker)
        if idx != -1:
            cut = idx + len(marker)
            if best_cut is None or cut > best_cut:
                best_cut = cut
    if best_cut is not None:
        return answer_text[best_cut:].strip()

    lines = [ln.strip() for ln in answer_text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def normalize(text):
    """Lowercase and strip markdown, quotes, a trailing period and a leading article."""
    text = text.strip().lower()
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\.+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_last_number(text):
    """Last integer or float in a string, as a string, or None."""
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def is_numeric(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def numeric_relaxed_match(pred, gold, tolerance=NUMERIC_TOLERANCE):
    try:
        pred_val = float(pred)
        gold_val = float(gold)
    except (TypeError, ValueError):
        return False
    if gold_val == 0:
        return pred_val == 0
    return abs(pred_val - gold_val) / abs(gold_val) <= tolerance


def text_match(pred, gold):
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    if not gold_n:                      # an empty gold would match anything
        return False
    if pred_n == gold_n:
        return True
    if ALLOW_SUBSTRING_MATCH:
        return gold_n in pred_n or (("[" in gold) and pred_n in gold_n)
    return False


def evaluate(data, verbose=False):
    correct_answers = []
    correct = 0

    for item in data:
        raw_pred = extract_final_answer(item["answer"])
        raw_gt = item["gt_answer"]

        if is_numeric(raw_gt):
            pred = extract_last_number(raw_pred)
            gt = re.sub(r"[^\d\.\-]", "", raw_gt)
            matched = pred is not None and numeric_relaxed_match(pred, gt)
        else:
            pred, gt = raw_pred, raw_gt
            matched = text_match(pred, gt)

        if matched:
            correct += 1
            correct_answers.append(item)
        elif verbose:
            print(f"[x] {item['image_id']} | pred: {raw_pred!r} | gt: {raw_gt!r}")

    return correct, correct_answers


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions and compute relaxed accuracy.")
    parser.add_argument("--input", required=True, help="Path to the result JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file for correct answers")
    parser.add_argument("--verbose", action="store_true", help="Print every mismatch")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    correct, correct_answers = evaluate(data, verbose=args.verbose)
    total = len(data)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nRelaxed accuracy: {accuracy:.2f}% ({correct}/{total})")

    with open(args.output, "w") as f:
        json.dump(correct_answers, f, indent=2)


if __name__ == "__main__":
    main()
