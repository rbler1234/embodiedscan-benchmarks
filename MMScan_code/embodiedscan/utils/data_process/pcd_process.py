import os
import numpy as np
import torch
from tqdm import tqdm
import json
import threading
from utils.box_utils import is_inside_box
from utils.data_io import read_annotation_pickle,id_mapping
from utils.task_utils import task_split
from scipy.spatial.transform import Rotation as R

def load_pcd_data(input_dir, scene):
    """
       load the data we get in the pcd_generate step
    """
    pcd_file = os.path.join(input_dir, scene, "pc_infos.npy")
    pc_infos = np.load(pcd_file)
    nan_mask = np.isnan(pc_infos).any(axis=1)
    pc_infos = pc_infos[~nan_mask]
    pc = pc_infos[:, :3]
    color = pc_infos[:, 3:6]
    label = pc_infos[:, 6].astype(np.uint16) 

    return pc, color, label

def create_scene_pcd(input_dir, output_dir, scene, es_anno,TYPE2INT, overwrite=True):
    """
        process the pcd with the embodiedscan info
    """
    
    if es_anno is None:
        return None
    if len(es_anno["bboxes"]) <= 0:
        return None
    out_file_name = os.path.join(output_dir, f"{scene}.pth")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    
    pc, color, label = load_pcd_data(input_dir, scene)
    label = np.ones_like(label) * -100
    if np.isnan(pc).any() or np.isnan(color).any():
        print(f"nan detected in {scene}")
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    bboxes =  es_anno["bboxes"].reshape(-1, 9)
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    object_ids = es_anno["object_ids"]
    object_types = es_anno["object_types"] # str
    sorted_indices = sorted(enumerate(bboxes), key=lambda x: -np.prod(x[1][3:6])) # the larger the box, the smaller the index
    sorted_indices_list = [index for index, value in sorted_indices]

    bboxes = [bboxes[index] for index in sorted_indices_list]
    object_ids = [object_ids[index] for index in sorted_indices_list]
    object_types = [object_types[index] for index in sorted_indices_list]

    for box, obj_id, obj_type in zip(bboxes, object_ids, object_types):
        obj_type_id = TYPE2INT.get(obj_type, -1)
        center, size, euler = box[:3], box[3:6], box[6:]
        orientation=R.from_euler("zxy",euler,degrees = False).as_matrix().tolist()
        box_pc_mask = is_inside_box(pc, center, size, orientation)
        num_points_in_box = np.sum(box_pc_mask)
        instance_ids[box_pc_mask] = obj_id
        label[box_pc_mask] = obj_type_id

    out_data = (pc, color, label, instance_ids)
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    torch.save(out_data, out_file_name)
    return True


def create_all(tasks,thread_id):
    print({f"Thread {thread_id} processing ... "})
    for task in tqdm(tasks):
        create_scene_pcd(task[0], task[1], task[2], task[3],task[4])
    print({f"Thread {thread_id} finish ... "})

def main():
    
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--pcd_mp3d_path", default="/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/matterport3d/scans", type=str)
    parser.add_argument("--pcd_3rscan_path", default="/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/3rscan/scans", type=str)
    parser.add_argument("--pcd_scannet_path", default="/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/scannet/scans", type=str)
    parser.add_argument("--out_dir_path", default=f"/mnt/hwfile/OpenRobotLab/linjingli/mmscan/process_pcd", type=str)
    args = parser.parse_args()
    

    path_dict = {'matterport3d':args.pcd_mp3d_path,
                '3rscan':args.pcd_3rscan_path,
                'scannet':args.pcd_scannet_path}

    output_dir = args.out_dir_path
    es_anno_file = '/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/embodiedscan-split/embodiedscan-v1/embodiedscan_infos_val.pkl'
    
    numthreads = 1
    es_annotation = read_annotation_pickle(es_anno_file)
    TYPE2INT = np.load(es_anno_file, allow_pickle=True)["metainfo"]["categories"]
   
    all_split_scans = es_annotation.keys()
    
    tasks = []
    scan_name_matcher = id_mapping()
    for scan_name in all_split_scans:
        scan_name_trans = scan_name_matcher.forward(scan_name)
        for data_base_name in ["matterport3d", "3rscan", "scannet"]:
            if data_base_name in scan_name:
                input_dir = path_dict[data_base_name]
        tasks.append([input_dir, output_dir, scan_name_trans,es_annotation[scan_name],TYPE2INT])

    
    task_split_dict = task_split(tasks,numthreads)
    
    threads = []
    for thread_id in range(numthreads):
        thread = threading.Thread(target = create_all, args=(task_split_dict[thread_id], thread_id))
        threads.append(thread)
        

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()    
            
if __name__ == "__main__":
    main()
