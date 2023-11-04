# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Demo script that shows data loading and model inference.
'''
import hydra
import numpy as np
import torch

from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu


def load_and_predict(data_dir, cfg):
    data, meta_data = load_rgb_xyz(
        data_dir, cfg.data.robot_prob,
        cfg.data.world_coord, cfg.data.jitter_scale,
        cfg.data.grid_resolution, cfg.eval.surface_range
    )
    if 'object_label' in meta_data:
        data['task'] = 'place'
    else:
        data['task'] = 'pick'

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    inputs, xyz, seg = data['inputs'], data['points'], data['seg']
    obj_inputs = data['object_inputs']
    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
        'placements': [],
        'placement_confidence': [],
        'placement_contacts': []
    }
    for _ in range(cfg.eval.num_runs):
        pt_idx = sample_points(xyz, cfg.data.num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz[pt_idx]
        data['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        data['object_inputs'] = obj_inputs[pt_idx]
        data_batch = collate([data])
        to_gpu(data_batch)

        with torch.no_grad():
            model_ouputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_ouputs)
        for key in outputs:
            if 'place' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_ouputs[key][0])
                ]
            else:
                outputs[key].extend(model_ouputs[key][0])
    data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
    data['object_inputs'] = obj_inputs
    return data, outputs


@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg):
    data, outputs = load_and_predict(cfg.eval.data_dir, cfg)

    vis = create_visualizer()
    rgb = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb = (rgb.numpy() * 255).astype('uint8')
    xyz = data['points'].numpy()
    cam_pose = data['cam_pose'].double().numpy()
    make_frame(vis, 'camera', T=cam_pose)
    if not cfg.eval.world_coord:
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)

    if data['task'] == 'pick':
        for i, (grasps, conf, contacts, color) in enumerate(zip(
            outputs['grasps'],
            outputs['grasp_confidence'],
            outputs['grasp_contacts'],
            get_set_colors()
        )):
            print(f"object_{i:02d} has {grasps.shape[0]} grasps")
            conf = conf.numpy()
            conf_colors = (np.stack([
                1 - conf, conf, np.zeros_like(conf)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"object_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )
            grasps = grasps.numpy()
            if not cfg.eval.world_coord:
                grasps = cam_pose @ grasps
            for j, grasp in enumerate(grasps):
                visualize_grasp(
                    vis, f"object_{i:02d}/grasps/{j:03d}",
                    grasp, color, linewidth=0.2
                )
    elif data['task'] == 'place':
        ee_pose = data['ee_pose'].double().numpy()
        make_frame(vis, 'ee', T=ee_pose)
        obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
        obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
        obj_xyz = obj_xyz_ee @ ee_pose[:3, :3].T + ee_pose[:3, 3]
        obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
        obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
        visualize_pointcloud(vis, 'object', obj_xyz, obj_rgb, size=0.005)
        for i, (placements, conf, contacts) in enumerate(zip(
            outputs['placements'],
            outputs['placement_confidence'],
            outputs['placement_contacts'],
        )):
            print(f"orientation_{i:02d} has {placements.shape[0]} placements")
            conf = conf.numpy()
            conf_colors = (np.stack([
                1 - conf, conf, np.zeros_like(conf)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"orientation_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )
            placements = placements.numpy()
            if not cfg.eval.world_coord:
                placements = cam_pose @ placements
            visited = np.zeros((0, 3))
            for j, k in enumerate(np.random.permutation(placements.shape[0])):
                if visited.shape[0] > 0:
                    dist = np.sqrt((
                        (placements[k, :3, 3] - visited) ** 2
                    ).sum(axis=1))
                    if dist.min() < cfg.eval.placement_vis_radius:
                        continue
                visited = np.concatenate([visited, placements[k:k+1, :3, 3]])
                visualize_grasp(
                    vis, f"orientation_{i:02d}/placements/{j:02d}/gripper",
                    placements[k], [0, 255, 0], linewidth=0.2
                )
                obj_xyz_placed = obj_xyz_ee @ placements[k, :3, :3].T \
                               + placements[k, :3, 3]
                visualize_pointcloud(
                    vis, f"orientation_{i:02d}/placements/{j:02d}/object",
                    obj_xyz_placed, obj_rgb, size=0.01
                )


if __name__ == '__main__':
    main()
