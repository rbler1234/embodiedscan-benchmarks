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
       
    """
    
    
    def __init__(self,eval_bs,model_config = model_config,max_length=1024, verbose=False) -> None:
        
        # eval_bs: batch size for evaluation
        
        # model_config: dict, contains the path of pth the special metric needed
        
        self.eval_bs = eval_bs
        self.verbose = verbose
        self. max_length = max_length
        self.special_metric = []
        
        
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

        self.metric_record = {metric:[] for metric in self.eval_metric}
        self.save_result = {}
        
    def save_results(self,path,is_main_process):
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

    def load_and_eval(self,batch_result):
        
        
        all_pred = self.to_mmscan_form(batch_result)
       
        self.save_result.update(all_pred)
        lan_gt = {}
        lan_pred = {}
        all_simcse_similarity = []
        all_sbert_similarity = []

        assert len(batch_result) <= self.eval_bs
                
        bar = all_pred if not self.verbose else tqdm(all_pred)

        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []

        for idx,key in enumerate(bar):
            pred = special_token_filter(all_pred[key]["pred"][0],clean=True,truncation=True,max_length=self.max_length)
            lan_pred[key] = [pred]
            
            lan_gt[key] = [special_token_filter(i,clean=True,truncation=True,max_length=self.max_length) for i in all_pred[key]["gt"]]
            batch_lan_pred += lan_pred[key]
            batch_lan_gt += lan_gt[key]
            count_gt += [len(lan_gt[key])]
        
        # (1) exact match evaluation
        EM_,refine_EM_ = self.EM_evaluation(lan_pred,lan_gt)
        # (2) sbert/simcse metric evaluation
        if "sbert" in self.special_metric:
            all_sbert_similarity = self.sbert_evaluation(batch_lan_pred,batch_lan_gt,count_gt)
        if "simcse" in self.special_metric:
            all_simcse_similarity = self.simcse_evaluation(batch_lan_pred,batch_lan_gt,count_gt)

        # (3) coco metric evaluation
        coco_scores,_ = self.coco_evaluate(ground_truths=lan_gt,prediction=lan_pred)


        eval_dict = {
            'EM':sum(EM_)/len(EM_),
            'refined_EM':sum(refine_EM_)/len(refine_EM_),
        }
        for metric_ in coco_scores:
            eval_dict[metric_] = coco_scores[metric_]
        if 'simcse' in self.special_metric:
            eval_dict['simcse'] = sum(all_simcse_similarity)/len(all_simcse_similarity),
        if 'sbert' in self.special_metric: 
            eval_dict['sbert'] = sum(all_sbert_similarity)/len(all_sbert_similarity),
        
        for metric in eval_dict:
            self.metric_record[metric].append(eval_dict[metric])
        return eval_dict

       
if __name__ == "__main__":
    
    import argparse
    eval_bs =10

    eval = Tradition_Evaluator(
        eval_bs=eval_bs
    )
    


        