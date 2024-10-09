# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/scannet_utils.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts
"""

import csv
from collections import defaultdict

import open3d as o3d
from plyfile import PlyData


def represents_int(s):
    """Judge whether string s represents an int.

    Args:
        s(str): The input string to be judged.

    Returns:
        bool: Whether s represents int or not.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename,
                       label_from='raw_category',
                       label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """Read XYZ for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        ndarray: Vertices.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def read_mesh_vertices_rgb(filename):
    """Read XYZ and RGB for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        Vertices. Note that RGB values are in 0-255.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices


# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load Scannet scenes with vertices and ground truth labels for semantic and
instance segmentations."""
import os
import json
import inspect
import argparse
import numpy as np

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    """Map object id to seg ids, label to seg ids."""
    assert os.path.isfile(filename)
    object_id_to_segs = defaultdict(list)
    label_to_segs = defaultdict(list)
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            label_to_segs[label].extend(segs)
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    """Map seg id to vert id."""
    assert os.path.isfile(filename)
    seg_to_verts = defaultdict(list)
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            seg_to_verts[seg_id].append(i)
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids,
                 num_neighbors = 100, std_ratio = .5):
    """
    Args:
        mesh_vertices: [N, 6], x y z r g b for each vertex.
        object_id_to_segs: dict map object id to segments.
        object_id_to_label_id: dict map object id to label id.
        instance_ids: [N], instance id for each vertex.
        num_neighbors: to remove outlier points.
        std_ratio: to remove outlier points.
    """
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    instance_bboxes = np.zeros((num_instances, 7))
    for i, obj_id in enumerate(object_id_to_segs.keys()):
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, :3]
        if len(obj_pc) == 0:
            continue

        # NOTE: Here we do the statistical_outlier_removal to remove
        # the outlier points, so that we get much tighter aabb.
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(obj_pc)
        pts, _ = pts.remove_statistical_outlier(num_neighbors, std_ratio)
        obj_pc = np.asarray(pts.points)        
        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)

        # Check if it is a opened door or a closed door if it is a door.
        # TODO: pass the args of closed door label id and open door label id
        if label_id == 8: # Original class: closed door
            xyz_center = (xyz_min + xyz_max) / 2.0
            xyz_eps_min = xyz_center - 0.1
            xyz_eps_max = xyz_center + 0.1
            xyz_door = mesh_vertices[instance_ids == obj_id, 0:3]
            if not ((xyz_door <= xyz_eps_max) & (xyz_door >= xyz_eps_min)).all(-1).any():
                label_id = 41 # Additional class: opened door

        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min, np.array([label_id])])
        # bbox = pull_out_bbox_inside_wall(bbox)
        
        # TODO: Check the thickness of the floor

        # NOTE: this assumes obj_id is in 1, 2, 3, ... NUM_INSTANCES
        # i == obj_id - 1 if there is no instances filtered out.
        instance_bboxes[i] = bbox
    return instance_bboxes


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None, test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        meta_file (str): Path of the meta_file.
        label_map_file (str): Path of the label_map_file.
        output_file (str): Path of the output folder.
            Default: None.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which contains the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """

    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, :3] = mesh_vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, :3], mesh_vertices[:, 3:]], axis=1)

    # Load semantic and instance labels
    if not test_mode:
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = defaultdict(int)
        # get label id for each vertex
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        # get instance id for each vertex
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                object_id_to_label_id[object_id] = label_ids[verts][0]
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids)
    else:
        label_ids = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None
        object_id_to_label_id = None

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        if not test_mode:
            np.save(output_file + '_sem_label.npy', label_ids)
            np.save(output_file + '_ins_label.npy', instance_ids)
            np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)
            np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, aligned_bboxes, object_id_to_label_id, axis_align_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scan_path',
        required=True,
        help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument(
        '--label_map_file',
        required=True,
        help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path,
                            scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(
        opt.scan_path, scan_name +
        '.txt')  # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file,
           opt.output_file)


if __name__ == '__main__':
    main()