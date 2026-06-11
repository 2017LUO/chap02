import time
from LLM_Agent import decide

if __name__ == "__main__":
    total_start = time.perf_counter()

    for code in range(6):
        t0 = time.perf_counter()
        line1, line2 = decide(code)          # 原始两行字符串
        single_elapsed = time.perf_counter() - t0

        print(f"[code {code}]  {single_elapsed*1000:.1f} ms")
        print("skills:", line1)   # list
        print("best  :", line2)     # int
        print(type(line2))
        print("-" * 40)

    total_elapsed = time.perf_counter() - total_start
    print(f"Total elapsed: {total_elapsed:.2f} s")