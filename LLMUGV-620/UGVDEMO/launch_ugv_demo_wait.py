import json
import os
import time
import numpy as np

from ugv_wrapper.wrapper_interface.ugv_wrapper import UGV

READY_FILE = r"F:\P\paper\code\LLMUGV-620\UGVDEMO\scene_ready.json"
TARGET_SCENE_NAME = "UGVParking"
UNITY_SCENE_ID = "scene_1"   # 你现有 wrapper 里使用的 scene 标识
POLL_INTERVAL = 0.2


def wait_for_scene_ready():
    print("[Python] waiting for scene_ready.json ...")
    while True:
        if os.path.exists(READY_FILE):
            try:
                with open(READY_FILE, "r", encoding="utf-8") as f:
                    info = json.load(f)

                if info.get("ready") and info.get("scene") == TARGET_SCENE_NAME:
                    print("[Python] Scene ready detected:", info)
                    return info
            except Exception as e:
                print("[Python] read ready file failed:", e)

        time.sleep(POLL_INTERVAL)


def main():
    info = wait_for_scene_ready()

    # 关键：env_name=None，表示连接 Unity Editor，而不是启动 exe
    env = UGV(
        seed=42,
        train_mode=False,
        env_name=None,
        n_envs=1,
        group_aggregation=True,
        scene=UNITY_SCENE_ID
    )

    try:
        image_stack, image, ray = env.reset()
        print("[Python] connected to Unity Editor successfully.")
        print("[Python] method =", info.get("method", "UNKNOWN"))

        # 这里先给你一个最小可运行控制循环
        # action_dim=2，先用全 0 动作做 smoke test
        action = np.zeros((1, 2), dtype=np.float32)

        for step in range(200):
            image_stack, image, ray, reward, done, pos_x, pos_z = env.step(action)

            print(f"[Python] step={step:03d} reward={reward:.3f} done={done} pos=({pos_x:.2f}, {pos_z:.2f})")

            if done:
                print("[Python] episode finished.")
                break

    finally:
        env.close()
        print("[Python] env closed.")


if __name__ == "__main__":
    main()