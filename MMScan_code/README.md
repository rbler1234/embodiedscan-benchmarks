# EMbodiedScan benchmark code

## 🏠 About

This codebase consist of: 

1. EmbodiedScan api, dataloader and evaluator.
2. Implement of some models



## 📚 EmbodiedScan api

### Tutorials

You can import embodiedscan api in this way:
```
# python code

    sys.path.append('path/to/embodiedscan/package')
    embodiedScan = importlib.import_module('embodiedscan')
    MMScan_eval = importlib.import_module('eval.tradition_evaluation')
```

#### DataLoader 

We provide a tools to get data samples from the embodiedScan/MMScan dataset.

1. Define a dataloader class (inherits from embodiedscan class), define your own data parsing function. 

```
    class MMScanDataLoader(embodiedScan.EmbodiedScan):

        def __init__(self, cfg, version='v1', split='train', verbose=True):
        ...
            super(self,MMScanDataLoader).__init__(dataroot=path/to/your/data ,version=version, split=split_set, verbose=verbose)
        ...

        def parse_dict(self,data_dict)->dict:
            ## TODO: define your own data parsing function
```
The `data_dict` we provide is in a dict format (for each sample):
``` 
    {
        # scan-level, info of the scan the sample belongs to
            
        "pcds": 3d point clouds (n, 6)
        "obj_pcds" : {object_id: pcds of object_id}
        "bboxes"
        "images"
        
        # anno-level
        ... : the same as the MMScan sample
    }          
```
2. Setting the lang task.

```
    loader = MMScanDataLoader(...)
    loader.set_lang_task("MMScan-QA")
```


#### Evaluator

1. Define a evaluator class (inherits from embodiedscan class), define your own changing form function. 

```
    class MMScanEvaluator(MMScan_eval.Tradition_Evaluator):
    
        def __init__(self, eval_bs=400,model_config = model_config,max_length=1024, verbose=False):
            super(MMScanEvaluator, self).__init__(eval_bs,model_config,max_length,verbose)
           
        def to_mmscan_form(self,raw_batch_result):
            ## TODO: define your own changing form function
```
The expect form is:
``` 
    {
        "pred": a list,
        "gt": a list,
        "":
    }          
```
2. Update the results every batch and record at the final.

```
    evaluator = MMScanEvaluator(...)
    for ...
        evaluator.update_results(raw_batch_result)
    metric_results = evaluator.start_evaluation()

    
    metric_results_record = evaluator.eval_results_record
```


### Some examples

You can explore the codes in `./models/LL3DA/datasets/unified_embodied_scan_qa.py` and `./models/LEO/data/datasets.py` to know more about our dataLoader api; explore the codes in `./models/LL3DA/eval_utils/evaluate_mmscan.py` and `./models/LEO/evaluator/mmscan_eval.py` to know more about our evaluator api;
