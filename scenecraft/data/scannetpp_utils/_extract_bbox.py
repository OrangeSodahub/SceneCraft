import os
import json
import argparse
from tqdm import tqdm
from scenecraft.renderer import ScannetppDataset


def main(args):
    # TODO: add bbox adjustment code
    os.makedirs(args.output_path, exist_ok=True)
    data_dir = os.path.join(args.data_root, 'data')
    all_scenes = list(sorted(os.listdir(data_dir)))
    print(f"======> Load {len(all_scenes)} scenes.")
    cat2label = {cat: ScannetppDataset.CLASSES.index(cat) for cat in ScannetppDataset.CLASSES}
    for scene in tqdm(all_scenes, leave=False):
        if scene == "preview":
            continue
        save_path = os.path.join(args.output_path, f"{scene}.json")
        json_path = os.path.join(data_dir, scene, 'scans', 'segments_anno.json')
        segments_anno = json.load(open(json_path, 'r'))
        segGroups = segments_anno["segGroups"]
        bboxes, labels = [], []
        for seg in tqdm(segGroups, leave=False):
            obb = seg['obb']
            centroid = obb["centroid"]
            bbox_min = obb["min"]
            bbox_max = obb["max"]
            dims = list(map(lambda min, max: max - min, bbox_min, bbox_max))
            label = cat2label.get(seg['label'], -1)
            if label == -1:
                continue
            bboxes.append(centroid + dims)
            labels.append(str(label))
        data = dict(bboxes=bboxes, labels=labels)
        with open(save_path, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, default="./data/scannetpp", required=False, help="Path to local dataset.")
    parser.add_argument(
        "--output-path", type=str, default="../scannetpp_processed/nerfstudio_aabb", required=False)
    args = parser.parse_args()
    main(args)
