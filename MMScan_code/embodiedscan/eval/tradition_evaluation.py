import argparse
import json
from collections import defaultdict
import re
import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm


from copy import deepcopy
from collections import OrderedDict

import re

# define some eval tools
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from scipy.spatial.distance import cosine
from glob import glob
import pickle
import copy
from simcse import SimCSE
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from eval.lang_utils import normalize_answer, special_token_filter ,exact_match_score

model_config = {"simcse":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/pc","sbert":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/st"}

class Tradition_Evaluator():
    
    """
    
       tradition metrics for QA and Caption evaluation , consists the implements of [EM, BLEU, METEOR, ROUGE, CIDEr, SPICE, SIMCSE, SBERT]
       
       main function:
       
       (1) reset() -> clear the buffer
       (2) update() -> update samples to the buffer and return EM & Refine_EM
       (3) start_evaluation() -> evaluate all samples in the buffer, return all metrics
        
    """
    
    
    def __init__(self,eval_bs=500,model_config = model_config,max_length=1024, verbose=False) -> None:
        
        # eval_bs: batch size for special model only 
        
        # model_config: dict, contains the path of pth the special metric needed
        
        self.eval_bs = eval_bs
        self.verbose = verbose
        self. max_length = max_length
        self.special_metric = []
        print(self.verbose)
        
        if "simcse" in model_config:
            self.special_metric.append("simcse")
            self.simcse_tokenizer = AutoTokenizer.from_pretrained(model_config["simcse"])
            self.simcse_model = AutoModel.from_pretrained(model_config["simcse"]).to("cuda")
        if "sbert" in model_config:
            self.special_metric.append("sbert")
            self.sbert_model = SentenceTransformer(model_config["sbert"],device="cuda")
            
        self.reset()

    def reset(self):
        self.eval_metric = ["EM","refined_EM","Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4","METEOR","ROUGE_L","CIDEr","SPICE"] + self.special_metric
        self.metric_record = {}
        self.eval_results_record ={}
        self.save_results = {}
        self.save_buffer = {'lan_pred':{},'lan_gt':{}}
        
    def save_json(self,path,is_main_process):
        assert path.endswith(".json"),"please provide a json file path"
        
        if is_main_process:
            with open(path,"w") as f:
                json.dump(self.save_result,f,indent=4)
            
            
    
    def to_mmscan_form(self,raw_batch_result):
        """
           add the transform from the original format to the expected format.
        """
        
        return raw_batch_result
    
    @staticmethod
    def to_coco(kvs, keys):
        res = defaultdict(list)
        for k in keys:
            if k in kvs:
                caps = kvs[k]
                for c in caps:
                    res[k].append({'caption': c})
            else:
                res[k].append({'caption': ''})
        return res

    def coco_evaluate(self,ground_truths,prediction):

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
        tokenizer = PTBTokenizer()
        ref_sent = ground_truths
        hypo_sent = prediction
        final_scores = {}
        final_list = {}
        ref_coco = tokenizer.tokenize(self.to_coco(ref_sent, ref_sent.keys()))
        hypo_coco = tokenizer.tokenize(self.to_coco(hypo_sent, ref_sent.keys()))
        for scorer, method in scorers:
            if self.verbose:
                print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(ref_coco, hypo_coco)
            if type(score) == list:
                for m, s, s_ in zip(method, score, scores):
                    final_scores[m] = s
                    final_list[m] = s_
            else:
                final_scores[method] = score
                final_list[method] = scores
    
        return final_scores, final_list
    
    def EM_evaluation(self,all_pred,all_gt):
        
        
        # simple EM
        EM_result = []
        cnt = []
        for ins in all_pred:
            pred  = all_pred[ins][0]
            if pred in all_gt[ins]:
                EM_result.append(1)
            else:
                EM_result.append(0) 
        
        # simple refined EM
        refine_EM_result = []
        
        for ins in all_pred:
            pred  = all_pred[ins][0]
            cnt = []
            for gt in all_gt[ins]:
                cnt.append(exact_match_score(pred,gt)) 
            refine_EM_result.append(max(cnt))
        return EM_result, refine_EM_result


    def sbert_evaluation(self,all_pred,all_gt,gt_count):
        """
        Args:
            gt_count(list): stores number of possible answers to a question
            all_pred(list): all prediction
            all_gt(list): all ground truth,   len(all_gt)>=len(all_pred)

        Return:
            tuple: all_sbert_sim,all_simcse_sim
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(all_pred+all_gt,show_progress_bar=False,device="cuda")
            
        all_pred_sbert_embed = sbert_embeddings[:len_of_pred]
        all_gt_sbert_embed = sbert_embeddings[len_of_pred:]


        all_sbert_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            sbert_similarity = -100
            for j in range(accumulated,accumulated+gt_count[i]):
                sbert_similarity = max(sbert_similarity, util.cos_sim(all_pred_sbert_embed[i], 
                                                                        all_gt_sbert_embed[j])[0][0].item())
            all_sbert_sim.append(sbert_similarity)
            accumulated+=gt_count[i]
        torch.cuda.empty_cache()
        return all_sbert_sim
    
    def simcse_evaluation(self,all_pred,all_gt,gt_count):
        """
        Args:
            gt_count(list): stores number of possible answers to a question
            all_pred(list): all prediction
            all_gt(list): all ground truth,   len(all_gt)>=len(all_pred)

        Return:
            tuple: all_sbert_sim,all_simcse_sim
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            inputs = self.simcse_tokenizer(all_pred+all_gt, padding=True, truncation=True, return_tensors="pt").to("cuda")
            simcse_embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        all_pred_simcse_embed = simcse_embeddings[:len_of_pred]
        all_gt_simcse_embed = simcse_embeddings[len_of_pred:]

        all_sbert_sim = []
        all_simcse_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            simcse_similarity = -100
            for j in range(accumulated,accumulated+gt_count[i]):
                simcse_similarity = max(simcse_similarity ,1 - cosine(all_pred_simcse_embed[i].cpu().detach().numpy(), 
                                                                        all_gt_simcse_embed[j].cpu().detach().numpy())) 
            all_simcse_sim.append(simcse_similarity)
            accumulated+=gt_count[i]
        torch.cuda.empty_cache()
        return all_simcse_sim

    
    def update(self,batch_result):
        
        all_pred = self.to_mmscan_form(batch_result)
        self.save_results.update(all_pred)
        lan_gt = {}
        lan_pred = {}
        bar = all_pred if not self.verbose else tqdm(all_pred)
        for _,key in enumerate(bar):
            pred = special_token_filter(all_pred[key]["pred"][0],clean=True,truncation=True,max_length=self.max_length)
            lan_pred[key] = [pred]
            lan_gt[key] = [special_token_filter(i,clean=True,truncation=True,max_length=self.max_length) for i in all_pred[key]["gt"]]
            
        self.save_buffer['lan_pred'].update(lan_pred)
        self.save_buffer['lan_gt'].update(lan_gt)
        
        EM_,refine_EM_ = self.EM_evaluation(lan_pred,lan_gt)
        return {"EM":sum(EM_)/len(EM_),"refined_EM":sum(refine_EM_)/len(refine_EM_)}
    
    def start_evaluation(self):
       
        assert len(self.save_results)>0
        assert len(self.save_buffer['lan_pred']) == len(self.save_buffer['lan_gt'])
        
        # (1) exact match evaluation
        EM_,refine_EM_ = self.EM_evaluation(self.save_buffer['lan_pred'],self.save_buffer['lan_gt'])
        
        # (2) coco metric evaluation
        coco_scores,coco_scores_list = self.coco_evaluate(ground_truths=self.save_buffer['lan_gt'],prediction=self.save_buffer['lan_pred'])
        
        
        # (3) special metric evaluation
        all_simcse_similarity = []
        all_sbert_similarity = []
        
        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []
        
        for idx, key in enumerate(self.save_buffer['lan_pred']):
            batch_lan_pred.extend(self.save_buffer['lan_pred'][key])
            batch_lan_gt.extend(self.save_buffer['lan_gt'][key])
            count_gt.extend([len(self.save_buffer['lan_gt'][key])])

            if (idx+1) % self.eval_bs == 0 or idx == len(self.save_buffer['lan_pred'])-1:
                
                if "sbert" in self.special_metric:
                    all_sbert_similarity += self.sbert_evaluation(batch_lan_pred,batch_lan_gt,count_gt)
                    
                if "simcse" in self.special_metric:
                    all_simcse_similarity += self.simcse_evaluation(batch_lan_pred,batch_lan_gt,count_gt)
                
                batch_lan_pred = []
                batch_lan_gt = []
                count_gt = []

       
        # collect the results, all metric results are in the same order as the ID list
      
        
        
        # (1) store for every sample, "Spice" only return the "all"
        store_dict = {'EM':EM_,'refined_EM':refine_EM_}
        
        for metric_ in coco_scores:
            if metric_ == "SPICE":
                store_dict[metric_] = [item["All"] for item in coco_scores_list["SPICE"]]
            else:    
                store_dict[metric_] = coco_scores_list[metric_]
        if 'simcse' in self.special_metric:
            store_dict['simcse'] = all_simcse_similarity
        if 'sbert' in self.special_metric: 
            store_dict['sbert'] = all_sbert_similarity
            
        
        for index, key in enumerate(self.save_buffer['lan_pred'].keys()):
            self.eval_results_record[key] = {}
            for metric in store_dict:
                self.eval_results_record[key][metric] = store_dict[metric][index]
        #print(self.eval_results_record)        

        
        # for metric in self.eval_metric:
        #     print(metric)
        #     print(store_dict[metric])
 
        
        
        # (2) return the final mean metric
    
        eval_dict = {}
        for metric in self.eval_metric:
            if metric != "SPICE":
     
                eval_dict.update({metric:sum(store_dict[metric])/len(store_dict[metric]) })
       
        eval_dict["SPICE"] = coco_scores["SPICE"]
        self.metric_record = eval_dict
        
       
        return eval_dict

    


        