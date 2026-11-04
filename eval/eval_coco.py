"""COCO caption evaluation, and export of the image ids kept for the analysis.

The exported ids are what the information-flow analysis runs on, so the CIDEr threshold
below decides which data enters the paper. 20 (i.e. CIDEr 0.20) is a permissive bar for
captioning; raise it to tighten the set.
"""

import argparse
import json

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO captions and save the ids above a CIDEr threshold."
    )
    parser.add_argument("--annotation", required=True,
                        help="Path to COCO annotations file (e.g., captions_val2014.json)")
    parser.add_argument("--results", required=True,
                        help="Path to the results JSON file with model captions")
    parser.add_argument("--ids_output", required=True,
                        help="Path to save selected image IDs JSON file")
    parser.add_argument("--cider_threshold", type=float, default=20.0,
                        help="CIDEr score threshold, in percent (default: 20.0)")
    parser.add_argument("--normalized_results", default=None,
                        help="Optional path to write the results with 'tokens_caption' renamed "
                             "to 'caption'. The input file is never modified in place.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the CIDEr score of every kept image")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        results = json.load(f)

    # pycocoevalcap expects the caption under 'caption'. The original script renamed the
    # key and wrote the result back over the input, destroying the raw model output; the
    # renamed list is all that loadRes needs, so keep it in memory.
    for res in results:
        if "tokens_caption" in res:
            res["caption"] = res.pop("tokens_caption")

    if args.normalized_results:
        with open(args.normalized_results, "w") as f:
            json.dump(results, f, indent=4)

    coco = COCO(args.annotation)
    coco_result = coco.loadRes(results)
    coco_eval = COCOEvalCap(coco, coco_result)

    # restrict the evaluation to the images actually present in the results
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {(score * 100):.3f}")

    image_ids_best = []
    missing = 0
    for img_id, eval_data in coco_eval.imgToEval.items():
        cider_raw = eval_data.get("CIDEr")
        if cider_raw is None:
            missing += 1
            continue
        cider_score = cider_raw * 100
        if cider_score > args.cider_threshold:
            image_ids_best.append(img_id)
            if args.verbose:
                print(f"Image ID {img_id}: CIDEr = {cider_score:.3f}")

    print(f"Kept {len(image_ids_best)} / {len(coco_eval.imgToEval)} images "
          f"above CIDEr {args.cider_threshold}")
    if missing:
        print(f"{missing} images had no CIDEr score and were skipped")

    with open(args.ids_output, "w") as f:
        json.dump(image_ids_best, f, indent=4)


if __name__ == "__main__":
    main()
