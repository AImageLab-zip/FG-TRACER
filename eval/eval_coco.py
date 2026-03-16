#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.cider.cider import Cider
import json
import argparse
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO captions and save best image IDs."
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to COCO annotations file (e.g., captions_val2014.json)",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the results JSON file with model captions",
    )
    parser.add_argument(
        "--ids_output",
        required=True,
        help="Path to save selected image IDs JSON file",
    )
    parser.add_argument(
        "--cider_threshold",
        type=float,
        default=20.0,
        help="CIDEr score threshold (in percent) for selecting best images (default: 20.0)",
    )

    args = parser.parse_args()

    annotation_file = args.annotation
    results_file = args.results
    ids_output_file = args.ids_output
    cider_threshold = args.cider_threshold

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load and normalize results
    with open(results_file, 'r') as f:
        results = json.load(f)

    for res in results:
        if 'tokens_caption' in res:
            res['caption'] = res.pop('tokens_caption')

    # Overwrite the same results file (as in your original script)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results)
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {(score * 100):.3f}')
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    image_ids_best = []
    max_score = 0
    for imgId, evalData in coco_eval.imgToEval.items():
        cider_raw = evalData.get('CIDEr', None)

        if cider_raw is not None:
            cider_score = cider_raw * 100  # convert to percent
            if cider_score > cider_threshold:
                image_ids_best.append(imgId)
                print(f"Image ID {imgId}: CIDEr Score = {cider_score:.3f}")
        else:
            print(f"Image ID {imgId}: CIDEr Score not available.")
    # %%########
    with open(ids_output_file, "w") as f:
        json.dump(image_ids_best, f, indent=4)
    # %%

if __name__ == "__main__":
    main()
