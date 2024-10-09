import argparse
import json
import os
from collections import defaultdict

import numpy as np

from scenecraft.data.scannet_utils._load_scannet_data import read_mesh_vertices_rgb, read_segmentation, extract_bbox


def read_aggregation(filename):
    """Map object id to seg ids, label to seg ids."""
    assert os.path.isfile(filename)
    object_id_to_segs = defaultdict(list)
    label_to_segs = defaultdict(list)
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] # instance ids in scannetpp is already 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            label_to_segs[label].extend(segs)
    return object_id_to_segs, label_to_segs


def export(mesh_file, agg_file, seg_file, label_map_file, output_file=None, test_mode=False,
           num_neighbors=100, std_ratio=.5):
    """Export original files to vert, ins_label, sem_label and bbox file.
    """

    # TODO: fix label_map after scannetpp release the mappings
    label_map = label_map_file
    # TODO: fix axis_align_matrxi after scannetpp release the axis alignments
    aligned_mesh_vertices = read_mesh_vertices_rgb(mesh_file)
    if output_file is not None:
        np.save(output_file + '_vert.npy', aligned_mesh_vertices)

    label_ids = None
    instance_ids = None
    aligned_bboxes = None
    object_id_to_label_id = None
    # Load semantic and instance labels
    if not test_mode:
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        # TODO: need to filter out the labels -1 after filling.
        label_ids = np.zeros(shape=(num_verts), dtype=np.int32) - 1 # -1: unannotated
        object_id_to_label_id = defaultdict(int)
        for label, segs in label_to_segs.items():
            # TODO: fix label map
            if label == 'split' or label == 'SPLIT':
                label = 'wall'
            if label not in label_map: continue
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(shape=(num_verts), dtype=np.int32) # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                if label_ids[verts][0] == -1: continue
                if instance_ids[verts][0] == 0:
                    instance_ids[verts] = object_id
                object_id_to_label_id[object_id] = label_ids[verts][0]

        # TODO: fix label map
        aligned_mesh_vertices = aligned_mesh_vertices[label_ids != -1]
        label_ids = label_ids[label_ids != -1]
        instance_ids = instance_ids[instance_ids != 0]
        object_id_to_segs = {k: v for k, v in object_id_to_segs.items() if k in object_id_to_label_id}
        # bboxes: [num_instances, 7]
        aligned_bboxes = extract_bbox(
            aligned_mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids, num_neighbors, std_ratio)

        if output_file is not None:
            np.save(output_file + '_sem_label.npy', label_ids)
            np.save(output_file + '_ins_label.npy', instance_ids)
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)

    return aligned_mesh_vertices, label_ids, instance_ids, object_id_to_label_id, aligned_bboxes


# For debug
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/scannetpp/data/0a7cc12c0e/')
    parser.add_argument('--output_file', required=True, help='output file')
    # TODO: remove this
    parser.add_argument('--label_map_file', required=True, help='')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(opt.scan_path, scan_name + '.txt')  # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file, opt.output_file)


if __name__ == '__main__':
    main()