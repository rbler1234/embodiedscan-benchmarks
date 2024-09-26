# embociedscan dev-kit.
# Code written by Jingli Lin, 2024.

import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable


import numpy as np
import torch

from tqdm import tqdm
from copy import deepcopy
import time

from utils.data_template import MMScan_QA_template
from utils.data_io import read_annotation_pickle,id_mapping,load_json
from utils.box_utils import __9DOF_to_6DOF__

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("MMScan dev-kit only supports Python version 3.")

class EmbodiedScan:
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1',
                 split: str = 'train',
                 dataroot: str = '/mnt/petrelfs/linjingli/mmscan_db/mmscan_data',
                 verbose: bool = True,
                 check_mode:bool = True,
                 ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version
        :param dataroot
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.split = split
        self.check_mode = check_mode
        if self.check_mode:
            print("embodiedscan's checking mode!!!")
        
        # initially the task is not defined
        self.task = None
        
        self.pkl_name = '{}/embodiedscan-split/embodiedscan-{}/embodiedscan_infos_{}.pkl'.format(self.dataroot, self.version, split)
        self.data_path = '{}/embodiedscan-split/data'.format(self.dataroot)
        self.info_path = '{}/embodiedscan-split/data_info'.format(self.dataroot)
        self.lang_anno_path = '{}/MMScan-beta-release'.format(self.dataroot)
        
        ## todo: prepare for the bash to generate the pcd. 
        self.pcd_path = '{}/embodiedscan-split/process_pcd'.format(self.dataroot)
        self.id_mapping = id_mapping()
        
        ##todo: deal with the test split
        self.table_names = ["point_clouds","bboxes","object_ids","object_types","object_type_ints","visible_view_object_dict","extrinsics_c2w","axis_align_matrix","intrinsics","depth_intrinsics","image_paths","depth_image_paths","visible_instance_ids"]
        self.lang_tasks = ["MMScan-QA","MMScan-VG","MMScan-DC","EmbodiedScan-VG","EmbodiedScan-Detection"]
        self.index_mode = "sample"

        assert osp.exists(self.pkl_name), 'Database version not found: {}'.format(self.pkl_name)

        if verbose:
            print("======\nLoading embodiedscan-{} database for split {}...".format(self.version,self.split))
            
        self.embodiedscan_anno = self.__load_base_anno__(self.pkl_name)
  
    def __getitem__(self, index_):
        assert self.task is not None, "Please set the task first!"
        
        data_dict = {}
        data_dict["index"] = index_
        
        if self.task == "MMScan-QA":
            
            
            # loading the data from the collection
            data_dict = {}
            scan_idx = self.MMScan_collect["anno"][index_]["scan_id"]
            data_dict["pcds"] = self.MMScan_collect["scan"][scan_idx]['pcds']
            data_dict["obj_pcds"] = self.MMScan_collect["scan"][scan_idx]['obj_pcds']
            data_dict["scene_center"] = self.MMScan_collect["scan"][scan_idx]['scene_center']
            data_dict["bboxes"] = self.MMScan_collect["scan"][scan_idx]['bboxes']
            data_dict["images"] = self.MMScan_collect["scan"][scan_idx]['images']
            data_dict.update(self.MMScan_collect["anno"][index_])
            
            # special processings for specific models
            self.parse_dict(data_dict)
            
        else:
            return None
    
    def __len__(self):
        assert self.task is not None, "Please set the task first!"
        return len(self.MMScan_collect["anno"])

    @property
    def show_possess(self) -> List[str]:
        """ Returns the tables that this class instance possesses. """
        return self.table_names
    @property
    def show_embodiedscan_id(self) -> List[str]:
        """ Returns the scan ids for this split. """
        return list(self.embodiedscan_anno.keys())
    @property
    def show_mmscan_id(self) -> List[str]:
        """ Returns the scan ids for this split. """
        assert self.task is not None, "Please set the task first!"
        return self.mmscan_scan_id
    
    @property
    def samples(self):
        """ Returns the scan ids for this split. """
        assert self.task is not None
        return self.mmscan_anno
    
    def set_lang_task(self, lang_task: str = "MMScan-QA"):
        """ Changing the mode to adapt the specific language task. """
        assert lang_task in self.lang_tasks, "{} is not found, we only support {}".format(lang_task, self.lang_tasks)
        self.task = lang_task
        if lang_task == "MMScan-QA":
            if self.verbose:
                print("==================\nNow the task is {}".format(self.task))
            
            start = time.time()
            self.mmscan_scan_id = load_json(f'{self.lang_anno_path}/Data_splits/{self.split}-split.json')
            self.mmscan_anno = load_json(f'{self.lang_anno_path}/MMScan_samples/MMScan_QA.json')[self.split]
            if self.check_mode:
                self.mmscan_anno = self.mmscan_anno[:20] # some samples to check
            
            # for more convenient use
            self.mmscan_anno_dict = {}
            for sample in self.mmscan_anno:
                
                if sample["scan_id"] not in self.mmscan_anno_dict:
                    self.mmscan_anno_dict[sample["scan_id"]] = []
                self.mmscan_anno_dict[sample["scan_id"]].append(sample)
            end = time.time()
            
            if self.verbose:
                print("==================\nLoading {} split for the {} task, using {} seconds".format(self.split,self.task,end-start))
            
            start = time.time()
            self.data_collect()
            end = time.time()
            
            if self.verbose:
                print("==================\nCollecting {} the data uses {} seconds".format(end-start))

    def get_possess(self, table_name: str, scan_idx: str=""):
        """ Returns the table with the given name if it exists. """
        assert table_name in self.table_names, "Table {} not found".format(table_name)
        if table_name == "point_clouds":
            if len(scan_idx)>0:
                return torch.load(f'{self.pcd_path}/{self.id_mapping.forward(scan_idx)}.pth')
            return {scan_idx: torch.load(f'{self.pcd_path}/{self.id_mapping(scan_idx)}.pth') for scan_idx in self.embodiedscan_anno.keys()}
        else:  
            if len(scan_idx)>0:
                return self.embodiedscan_anno[scan_idx][table_name]
            return {scan_idx: self.embodiedscan_anno[scan_idx][table_name] for scan_idx in self.embodiedscan_anno.keys()}
    
    def data_collect(self,flatten = True)->dict:
        """ Collect the useful data for the specific task """
       
        assert self.task is not None, "Please set the task first!"
        self.MMScan_collect = {}
        
        # (1) embodiedscan anno processing
        
        self.MMScan_collect["scan"] = {}
        for scan_idx in self.show_mmscan_id:
            self.MMScan_collect["scan"][scan_idx] = self.pcd_info_process(scan_idx) 
            self.MMScan_collect["scan"][scan_idx]['images'] = self.img_info_process(scan_idx)
            self.MMScan_collect["scan"][scan_idx]['bboxes'] = self.box_info_process(scan_idx)
        
        # (2) MMScan anno processing
        if self.task == "MMScan-QA":
            
            # for MMScan-QA task
            self.MMScan_collect["anno"] = []
            
            for sample in self.mmscan_anno:
                if self.split == 'train' and flatten:
                    for answer in sample["answers"]:
                        sub_sample = deepcopy(sample)
                        sub_sample['answers'] = [answer]
                        self.MMScan_collect["anno"].append(sub_sample)
                else:
                    self.MMScan_collect["anno"].append(sample)
                       
        else:
            assert 1==2, "not implemented yet"
        
        
        
    
    def parse_dict(self,data_dict)->dict:
        return data_dict
    
    
    # embodiedscan processing part
           
    def __load_base_anno__(self, pkl_path) -> dict:
        """ Loads the embodiedscan pkl file ( for a single scene or a split ) . """ 
        
        return read_annotation_pickle(pkl_path,self.verbose,show_progress=self.verbose)

    def pcd_info_process(self, scan_idx: str):
        """ help process the pcd in a specific form"""
        
        # Input scan id, return the scan info including pcd, obj_pcds, scene_center
        assert scan_idx in self.embodiedscan_anno.keys(), "Scan {} is not in {} split".format(scan_idx,self.split)
        scan_info = {}
        pcd_data = torch.load(f'{self.pcd_path}/{self.id_mapping.forward(scan_idx)}.pth')
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        
        pcds = np.concatenate([points, colors], 1)
        
        # (0) return the original pcds without process
        scan_info['ori_pcds'] = deepcopy(pcd_data)
        
        # (1) the point cloud with position [x,y,z], color [r,g,b], type label and object id 
        scan_info['pcds'] = deepcopy(pcds)
        
        obj_pcds = {}
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i
            if len(pcds[mask])>0:
                obj_pcds.update({i: pcds[mask]})

        # (2) the object point clouds dict
        scan_info['obj_pcds'] = obj_pcds
        
        # (3) the scene center
        scan_info['scene_center'] = (points.max(0) + points.min(0)) / 2

        return scan_info
    
    def box_info_process(self, scan_idx: str):
        """ help process the box in a specific form"""
        # Input scan id, return the box info of the object in a dict format
        # {id : {"bbox":np.array, "type":str}}
        assert scan_idx in self.embodiedscan_anno.keys(), "Scan {} is not in {} split".format(scan_idx,self.split)
        bboxes = self.get_possess("bboxes",scan_idx)
        object_ids = self.get_possess("object_ids",scan_idx)
        object_types = self.get_possess("object_types",scan_idx)
        return {object_ids[i]: {"bbox":bboxes[i], "type":object_types[i]} for i in range(len(object_ids))}
    
    def down_9DOF_to_6DOF(self, pcd: np.ndarray, box_9DOF: np.ndarray) -> np.ndarray:
        return __9DOF_to_6DOF__(pcd, box_9DOF)
    
    def img_info_process(self, scan_idx: str):
        """ help process the images info in a specific form"""
        assert scan_idx in self.embodiedscan_anno.keys(), "Scan {} is not in {} split".format(scan_idx,self.split)
        
        img_info = dict()
        img_info['img_paths'] = self.get_possess("image_paths",scan_idx)
        img_info['depth_image_paths'] = self.get_possess("depth_image_paths",scan_idx)
        img_info['intrinsics'] = self.get_possess("intrinsics",scan_idx)
        img_info['depth_intrinsics'] = self.get_possess("depth_intrinsics",scan_idx)
        img_info['extrinsics_c2w'] = self.get_possess("extrinsics_c2w",scan_idx)
        img_info['visible_instance_ids'] = self.get_possess("visible_instance_ids",scan_idx)
        
        img_info_list = []
        for camera_index in range(len(img_info["img_paths"])):
            item = {}
            for possess in img_info.keys():
                item[possess] = img_info[possess][camera_index]
            img_info_list.append(item)
        return img_info_list
    
  

if __name__ =="__main__":
    test = EmbodiedScan(version='v1', split='val', verbose=True)
    test.set_lang_task()
    print(len(test))
    print(test[100])