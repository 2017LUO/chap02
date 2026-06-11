import textwrap

delimiter = "####"

# ────────── Few-shot example (messages & answer) ────────── #
example_message = textwrap.dedent(f"""\
{delimiter} Driving scenario description:
You are driving in an environment that consists of both a racetrack section and a street section.
You begin on the racetrack, where you will encounter multiple subtasks that require switching skills.
Afterwards, you will proceed to the street environment under the guidance of traffic lights and signs.
Currently, you are in the racetrack environment, and there is a green obstacle ahead.

{delimiter} Your available skills:
Lane Keeping       - Continue driving in the current lane without switching to another skill      Skill_id: 0
Obstacle Avoidance - Detect and avoid green obstacles while maintaining normal navigation         Skill_id: 1
Target Collision   - Intentionally collide with the yellow sphere                                 Skill_id: 2
Street Entry       - Upon encountering traffic lights and signs, transition onto the main street  Skill_id: 3
Reaching Parking   - After entering the street, locate a suitable parking area                    Skill_id: 4
Parking            - Upon finding the parking zone, execute a complete stop                       Skill_id: 5
""")

example_answer = textwrap.dedent(f"""\
Well, I have 6 skills to choose from. Now I need to first infer which four skills fit the current scenario,
then determine the single best skill to complete the task.
According to the current scenario:
1. I'm still on the racetrack section (no traffic lights or signs mentioned yet), so Street Entry (Skill_id: 3) might be relevant soon but not immediately.
2. There's a green obstacle on the track, so Obstacle Avoidance (Skill_id: 1) seems crucial right now.
3. Lane Keeping (Skill_id: 0) could be a fallback, but it doesn't address the obstacle issue.
4. Target Collision (Skill_id: 2) is meant for a yellow sphere, which hasn't appeared, so it's not relevant at this moment.
5. Reaching Parking (Skill_id: 4) and Parking (Skill_id: 5) are skills for the street section, which we haven’t entered yet.

Hence, the four skills that might be considered here are:
- Lane Keeping (0)
- Obstacle Avoidance (1)
- Target Collision (2)
- Street Entry (3)

Because there is a green obstacle directly ahead, the best choice is **Obstacle Avoidance** (Skill_id: 1).

Final Answer: Obstacle Avoidance

Response to user:
#### 0 1 2 3
#### 1
""")

# ────────── System instruction ────────── #
system_message = textwrap.dedent(f"""\
You are a large language model. Now you act as a mature driving assistant, who first thinks of four applicable skills
that fit the current scenario and then selects the best one to accomplish the task at hand in a long-horizon, complex UGV control task.
You will be given a detailed description of the driving scenario of current frame along with your history of previous
decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited
by {delimiter}.

Your response should use the following format:
<reasoning>
<reasoning>
<repeat until you have a decision>
Response to user:
{delimiter} <First line: output exactly four Skill_id integers (space-separated) that are most
relevant to the current scenario.>
{delimiter} <second line: output the single Skill_id of the best skill to solve the task.>
Only numbers—no skill names or explanations.

Make sure to include {delimiter} to separate every step, **and do NOT output any reasoning—only the two-line numeric result**
""")

# ────────── Human message template ────────── #
human_message_template = textwrap.dedent(f"""\
Above messages are some examples of how you make a decision successfully in the past.
Those scenarios are similar to the current scenario.
You should refer to those examples to make a decision for the current scenario.

Here is the current scenario:
{delimiter} Driving scenario description:
{{scenario_description}}
{delimiter} Driving Intensions:
{{driving_intensions}}
{delimiter} Available skills:
{{available_skills}}

You can stop reasoning once you have a valid action to take.
""")


# ────────── Prompt assembly helper ────────── #
def assemble_full_prompt(scenario_description: str,
                         driving_intensions: str,
                         available_skills: str) -> str:
    """
    Combine the system message, few-shot examples, and current scenario
    into one clean prompt string.
    """
    human_message = human_message_template.format(
        scenario_description=scenario_description,
        driving_intensions=driving_intensions,
        available_skills=available_skills
    )

    final_prompt = textwrap.dedent(f"""{system_message}

[Human, few-shot #1]
{example_message}

[AI, few-shot #1]
{example_answer}

[Human, current scenario]
{human_message}
""")

    return final_prompt.strip()   # remove leading/trailing blank lines


# ────────── Demo run ────────── #
#There's a green obstacle ahead.
if __name__ == "__main__":
    current_scene = textwrap.dedent("""\
    You are still on the racetrack, moving at a moderate speed.
    
    There's a yellow sphere ahead.
    """)
    current_intentions = ("The driver aims to drive safely while avoiding green obstacles or intentionally colliding "
                          "with high-value yellow spheres, and ultimately complete a series of tasks to park safely in "
                          "a designated street parking zone.")
    current_skills = textwrap.dedent("""\
    Lane Keeping       - Continue driving in the current lane without switching to another skill      Skill_id: 0
    Obstacle Avoidance - Detect and avoid green obstacles while maintaining normal navigation         Skill_id: 1
    Target Collision   - Intentionally collide with the yellow sphere                                 Skill_id: 2
    Street Entry       - Upon encountering traffic lights and signs, transition onto the main street  Skill_id: 3
    Reaching Parking   - After entering the street, locate a suitable parking area                    Skill_id: 4
    Parking            - Upon finding the parking zone, execute a complete stop                       Skill_id: 5
    """)

    prompt_str = assemble_full_prompt(
        scenario_description=current_scene,
        driving_intensions=current_intentions,
        available_skills=current_skills
    )

    print(prompt_str)
