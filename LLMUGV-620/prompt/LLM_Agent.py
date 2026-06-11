import requests
from typing import Tuple, List

from prompt.UGV_Prompt import assemble_full_prompt

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # 本地 Ollama 服务
MODEL = "llama3.3"
DELIM = "####"

# 0-5 场景补充句（已翻译成英语）
SCENE_SUFFIX = {
    0: "There is nothing on the road ahead; keep your course.",
    1: "There is a green obstacle ahead on the road.",
    2: "There is a yellow sphere ahead on the road.",
    3: "There are traffic lights and direction signs ahead.",
    4: "You have already entered the street environment.",
    5: "You have located the designated parking zone ahead.",
}

# 其他固定文本（跟你原脚本里的 current_intentions / current_skills 一样）
INTENTIONS = (
    "The driver aims to drive safely while avoiding green obstacles or "
    "intentionally colliding with high-value yellow spheres, and ultimately "
    "complete a series of tasks to park safely in a designated street zone."
)

AVAILABLE_SKILLS = """\
Lane Keeping       - Continue driving in the current lane without switching to another skill      Skill_id: 0
Obstacle Avoidance - Detect and avoid green obstacles while maintaining normal navigation         Skill_id: 1
Target Collision   - Intentionally collide with the yellow sphere                                 Skill_id: 2
Street Entry       - Upon encountering traffic lights and signs, transition onto the main street  Skill_id: 3
Reaching Parking   - After entering the street, locate a suitable parking area                    Skill_id: 4
Parking            - Upon finding the parking zone, execute a complete stop                       Skill_id: 5
"""


def _scene_text(code: int) -> str:
    if code not in SCENE_SUFFIX:
        raise ValueError("code must be 0-5")
    return (
        SCENE_SUFFIX[code]
    )


def _call_ollama(prompt: str) -> Tuple[List[int], int]:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=100000)
    r.raise_for_status()
    raw = r.json()["response"].strip()
    lines = [ln for ln in raw.splitlines() if ln.startswith(DELIM)]
    if len(lines) != 2:
        raise RuntimeError("Unexpected LLM output:\n" + raw)

    line1, line2 = lines[0], lines[1]  # 原始两行字符串

    # ───── 解析数字 ─────
    skills = [int(x) for x in line1.split()[1:]]  # → [0, 1, 2, 3]
    best = int(line2.split()[1])  # → 0 / 1 / …
    return skills, best   # 第一行、第二行


def decide(code: int) -> Tuple[List[int], int]:
    """
    输入 0-5，返回 (line1, line2)，两行都带 '####' 前缀。
    - line1：4 个技能号
    - line2：最佳技能号
    """
    prompt = assemble_full_prompt(
        scenario_description=_scene_text(code),
        driving_intensions=INTENTIONS,
        available_skills=AVAILABLE_SKILLS,
    )
    return _call_ollama(prompt)
