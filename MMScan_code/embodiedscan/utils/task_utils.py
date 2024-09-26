import os
def task_split(tasks, numthreads):
    assigned_tasks = {}
    for _index, _task in enumerate(tasks):
        if _index % numthreads not in assigned_tasks:
            assigned_tasks[_index % numthreads] = []
        assigned_tasks[_index % numthreads].append(_task)
    return assigned_tasks
        
    
