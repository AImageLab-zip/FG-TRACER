import json
import re
import argparse

correct_answers = []

def extract_last_number(text):
    """
    Extracts the last number (int or float) from a string.
    """
    text = text.replace(',', '')
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)  # Finds all ints/floats
    if matches:
        return matches[-1]
    return None

def extract_final_answer(answer_text): 
    if "Answer:**" in answer_text:
        return answer_text.split("Answer:**")[-1].strip()
    elif "Answer*:" in answer_text:
        return answer_text.split("Answer*:")[-1].strip()
    elif "Answer**:" in answer_text:
        return answer_text.split("Answer**:")[-1].strip()
    elif "Answer:" in answer_text:
        return answer_text.split("Answer:")[-1].strip()
    elif "**FINAL ANSWER:**" in answer_text:
        return answer_text.split("**FINAL ANSWER:**")[-1].strip()
    elif "FINAL ANSWER:" in answer_text:
        return answer_text.split("FINAL ANSWER:")[-1].strip()
    return answer_text.strip().split('\n')[-1].strip()

def is_numeric(value):
    try:
        float(value)
        return True
    except:
        return False

def numeric_relaxed_match(pred, gold):
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) / abs(gold_val) <= 0.05
    except:
        return False

def exact_match(pred, gold):
    if '[' in gold:
        return pred.strip().lower() in gold.strip().lower()
    else:
        return pred.strip().lower() == gold.strip().lower() or gold.strip().lower() in pred.strip().lower()

correct = 0
total = 0

word_to_num = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

def estrai_numero(text):
    """
    Estrae il primo numero decimale o intero da una stringa.
    """
    match = re.search(r'[-+]?[0-9]*\.?[0-9]+', text)
    if match:
        return float(match.group())
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions and compute relaxed accuracy.")
    parser.add_argument("--input", required=True, help="Path to the result JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file for correct answers")

    args = parser.parse_args()

    global correct, total

    with open(args.input, "r") as f:
        data = json.load(f)

    for item in data:
        raw_pred = extract_final_answer(item["answer"])
        raw_gt = item["gt_answer"]

        if is_numeric(raw_gt):
            pred = extract_last_number(raw_pred)
            if not pred:
                matched = False
            else:
                pred = pred.lower()
                while (pred[-1] == '.'):
                    pred = pred[:-1]

                gt = re.sub(r'[^\d\.\-]', '', raw_gt)

                # number in string -> number in digits
                if not is_numeric(pred) and pred in word_to_num.keys():
                    pred = word_to_num.get(pred.lower(), None)

                matched = numeric_relaxed_match(pred, gt)
        else:
            pred = raw_pred
            gt = raw_gt
            if pred and pred[-1] == '.':
                pred = pred[:-1]

            matched = exact_match(pred, gt)

        if matched:
            correct += 1
            correct_answers.append(item)
        else:
            print(f"[❌] {item['image_id']} | Raw Pred: {raw_pred} | Pred: {pred} | GT: {gt}")

        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n✅ Relaxed Accuracy: {accuracy:.2f}% ({correct}/{total})")

    with open(args.output, "w") as f:
        json.dump(correct_answers, f, indent=2)


if __name__ == "__main__":
    main()
