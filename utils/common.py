from utils import Logging

def get_task(str_task):
    task_category = ["rename", "align"]
    tasks = str_task.split(",")
    flag = False
    for task in tasks:
        for tc in task_category:
            if task == tc:
                flag = True
    assert flag, Logging.e("\"{}\" task is incorrect.".format(task))
    return tasks