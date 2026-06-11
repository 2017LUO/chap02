import time
import random
from typing import List, Tuple


def decide(task_id: int) -> Tuple[List[int], int]:
    """
    Parameters
    ----------
    task_id : int
        子任务编号 (0–5)

    Returns
    -------
    fixed_skills : List[int]
        根据 task_id 返回对应的技能列表:
        * 0, 1, 2 → [0, 1, 2, 3]
        * 3, 4, 5 → [2, 3, 4, 5]
    cur_skill : int
        与 task_id 相同
    """
    if task_id in (0, 1, 2):
        fixed_skills: List[int] = [0, 1, 2, 3]
    elif task_id in (3, 4, 5):
        fixed_skills = [2, 3, 4, 5]
    else:
        raise ValueError("task_id must be between 0 and 5")

    cur_skill = task_id
    # 随机等待 0.91–1.00 秒，避免过于规律的调用
    time.sleep(random.uniform(0.91, 1.09))
    return fixed_skills, cur_skill

