"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannetpp_data.py
"""
import argparse
import datetime
import os
from os import path as osp

import numpy as np
from tqdm import tqdm

from scenecraft.data.scannetpp_utils._load_scannetpp_data import export
from scenecraft.data.dataset import ScannetppDataset

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.arange(len(ScannetppDataset.CLASSES))


def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannetpp_dir,
                    test_mode=False):
    mesh_file = osp.join(scannetpp_dir, scan_name, 'scans', 'mesh_aligned_0.05.ply')
    agg_file = osp.join(scannetpp_dir, scan_name, 'scans', 'segments_anno.json')
    seg_file = osp.join(scannetpp_dir, scan_name, 'scans', 'segments.json')
    mesh_vertices, semantic_labels, instance_labels, instance2semantic, aligned_bboxes = export(
            mesh_file, agg_file, seg_file, label_map_file, None, test_mode, num_neighbors=150, std_ratio=.5)

    if not test_mode:
        mask = np.logical_not(np.isin(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        bbox_mask = np.isin(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        print(f'Num of care instances: {aligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    if not test_mode:
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)


def batch_export(max_num_point, output_folder, data_root, label_map_file, test_mode=False):
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.makedirs(output_folder, exist_ok=True)

    scan_names = list(os.listdir(data_root))
    label_map_file = ScannetppDataset.cat2label
    for scan_name in tqdm(scan_names):
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(f"Processing scene: {scan_name}")
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            label_map_file, data_root, test_mode)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num_point', default=1000000, help='The maximum number of the points.')
    parser.add_argument('--output_folder', default='./data/scannetpp_processed/scannetpp_instance_data', help='output folder of the result.')
    parser.add_argument('--data_root', default='./data/scannetpp/data', help='scannetpp data directory.')
    parser.add_argument(
        '--label_map_file', default='./proj/data/scannetpp_utils/meta_data/scannetv2-labels.combined.tsv', help='The path of label map file.')
    args = parser.parse_args()
    batch_export(args.max_num_point, args.output_folder, args.data_root, args.label_map_file, test_mode=False)


if __name__ == '__main__':
    main()