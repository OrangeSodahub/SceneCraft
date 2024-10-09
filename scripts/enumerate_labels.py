import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from scenecraft.data.scannetpp_utils._load_scannetpp_data import read_aggregation
from scenecraft.data.scannet_utils._load_scannet_data import read_segmentation


# scannetpp
def main(args):
    scan_names = list(os.listdir(args.data_root))
    num_labels = defaultdict(lambda: [0, 0, 0])
    label_map_file = os.path.join(args.data_root, '../metadata/semantic/semantic_classes.txt')
    with open(label_map_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    label_map = [line.strip() for line in lines]

    label_to_label_id = {label: label_id for label_id, label in enumerate(label_map)}
    label_id_to_label = {label_id: label for label, label_id in label_to_label_id.items()}

    for scan_name in tqdm(scan_names):
        agg_file = os.path.join(args.data_root, scan_name, 'scans', 'segments_anno.json')
        seg_file = os.path.join(args.data_root, scan_name, 'scans', 'segments.json')

        try:
            # Load semantic and instance labels
            object_id_to_segs, label_to_segs = read_aggregation(agg_file)
            seg_to_verts, num_verts = read_segmentation(seg_file)
            label_ids = np.zeros(num_verts) - 1
            object_id_to_label_id = defaultdict(int)
            for label, segs in label_to_segs.items():
                for seg in segs:
                    verts = seg_to_verts[seg]
                    num_labels[label][0] += len(verts)
                    if label not in label_to_label_id:
                        label_to_label_id[label] = len(label_to_label_id)
                        label_id_to_label[len(label_to_label_id) - 1] = label
                    label_ids[verts] = label_to_label_id[label]

            for object_id, segs in object_id_to_segs.items():
                for seg in segs:
                    verts = seg_to_verts[seg]
                    if label_ids[verts][0] == "none": continue
                    object_id_to_label_id[object_id] = label_ids[verts][0]

        except Exception as e:
            print(f"Skipped {scan_name} due to {e}.")

        for object_id, label_id in object_id_to_label_id.items():
            num_labels[label_id_to_label[label_id]][1] += 1

    for label in num_labels.keys():
        num_labels[label][2] = num_labels[label][0] // num_labels[label][1]    

    print(f"Scanned {len(scan_names)} scenes")
    num_labels = list(sorted(num_labels.items(), key=lambda x: x[1][0], reverse=True))
    with open(f"enumerate_labels.jsonl", "w") as f:
        f.write('\n'.join(map(json.dumps, num_labels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/scannetpp/data", required=False)
    args = parser.parse_args()
    main(args)