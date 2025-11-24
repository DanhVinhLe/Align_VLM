import json
import random

# Path tới file của bạn
path = "/workspace/ComfyUI/models/hypernetworks/MobileVLM/data/finetune_data/MobileVLM_V2_FT_Mix2M.json"

# Đọc file JSON
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy 100 mẫu ngẫu nhiên
samples = random.sample(data, 100)

# In ra
for i, s in enumerate(samples, 1):
    print(f"--- Sample {i} ---")
    print(s)
    print()
