import threading
import json
import random
import requests
import  os
from tqdm import tqdm

from openai import OpenAI

from eval.lang_utils import QA_prompt_define,QA_metric_map

class GPT_Evaluator():
    
    """
    
       GPT metric, we set this for QA and Caption tasks.
       
    """
    def __init__(self, eval_size=-1, API_key="", model='gpt-mini-4o',verbose = False):
        self.eval_size = eval_size
        self.API_key = API_key
        self.model = model
        self.verbose = verbose
        self.cilent = OpenAI(api_key=API_key)
        self.QA_metric = ['STa', 'STs', 'OOa', 'OOs', 'OR', 'overall', 'Advanced']

    def normal_query(self,system_prompt,user_content_groups,max_tokens=1000):
        
        # Send the system prompt and user content groups to GPT and get the response in a JSON format 
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for content_group in user_content_groups:
            messages.append({"role": "user", "content": content_group})

        response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens
                )
        response = json.loads(response.choices[0].message.content)
        return response  

    def QA_evaluation(self, QA_sample_dict, thread_index, tmp_path):
        
        '''
            multi-proc gpt-qa-evalation
        '''
        
        system_prompt, ex_instance = QA_prompt_define()
        
        # Define the number of retries
        MAXTRY = 3
        gpt_eval_results = {}
        
        for QA_ID in tqdm(QA_sample_dict):
            
            GPT_INTPUT = {"Question":QA_sample_dict[QA_ID]["question"],"Model Answer":QA_sample_dict[QA_ID]["pred"],"Human Answer":QA_sample_dict[QA_ID]["gt"][0]}
            
            for _ in range(MAXTRY):
                
                FLAG = False
                try:
                    GPT_OUTPUT = self.normal_query("gpt-4o-mini", system_prompt + ex_instance, [str(GPT_INTPUT)])
                    assert "All key points" in GPT_OUTPUT and "Correct Number" in GPT_OUTPUT and "Wrong/Missing Number" in GPT_OUTPUT and "Reasons" in GPT_OUTPUT
                    assert len(GPT_OUTPUT["All key points"])==int(GPT_OUTPUT["Correct Number"])+int(GPT_OUTPUT["Wrong/Missing Number"])
                    FLAG = True
                except:
                    continue
                if FLAG:
                    gpt_eval_results[QA_ID] = GPT_OUTPUT
                    
        with open(tmp_path.replace('.json','_thread'+str(thread_index)+'.json'), 'w') as f:
            json.dump(gpt_eval_results,f,indent=4)
    
    def QA_collection(self, num_threads, tmp_path):
        
        """
            collect the gpt-eval results
        """
        
        eval_dict = {metric:[] for metric in self.QA_metric}
        static_result = {}
        for thread_index in range(num_threads):
            with open(tmp_path.replace('.json','_thread'+str(thread_index)+'.json'), 'r') as f:
                thread_result = json.load(f)
            for ID in thread_result:
                static_result[ID] = thread_result[ID]
        for ID in static_result:
            eval_dict[QA_metric_map(ID.split('__')[0])].append(int(static_result["Correct Number"])/(int(static_result["Correct Number"])+int(static_result["Wrong/Missing Number"])))
        
        for metric in eval_dict:
            eval_dict[metric] = sum(eval_dict[metric])/len(eval_dict[metric])
        
        return eval_dict
            
    
    
    def load_and_eval(self, batch_result, num_threads=1, tmp_path = './'):
        
        if self.eval_size == -1:
            num_sample = len(batch_result)
        QA_sample = random.sample(list(batch_result.keys()),num_sample)   

        
        threads = []
        QA_IDs = list(QA_sample)
        IDs_divide_index = []
        for _index in range(num_threads):
            IDs_divide_index.append(QA_IDs[len(QA_IDs)//num_threads*_index:len(QA_IDs)//num_threads*(_index+1)])

        for thread_index in range(num_threads):
            
            # Create a sub-dictionary for each thread
            QA_sample_dict = {ID_: batch_result[ID_] for ID_ in IDs_divide_index[thread_index]}
            if self.verbose:
                print(f"Thread {thread_index} processing {len(QA_sample_dict)}")
            thread = threading.Thread(target = self.QA_evaluation, args=(QA_sample_dict, thread_index, tmp_path+'/gpt_QA.json'))
            threads.append(thread)

        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        print(f"Finished GPT evaluation, the results are store under {tmp_path}")
        
        eval_dict = self.QA_collection(num_threads, tmp_path)
        return eval_dict

    
    
    
