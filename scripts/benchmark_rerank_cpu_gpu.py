"""
重排模型 CPU vs GPU 速度对比测试
运行方式（在项目根目录 aicity 下）：
  python -m backend.scripts.benchmark_rerank_cpu_gpu
或：
  cd backend && python scripts/benchmark_rerank_cpu_gpu.py
"""
import os
import sys
import time

# 在导入 torch / transformers 之前设置
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 保证能导入 backend 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from sentence_transformers import CrossEncoder


def get_rerank_model_path():
    try:
        from backend.config import settings
        return settings.RERANK_MODEL_PATH or settings.RERANK_MODEL_NAME
    except Exception:
        return "BAAI/bge-reranker-large"


def load_model(device: str):
    model_path = get_rerank_model_path()
    kwargs = {"device": device}
    try:
        return CrossEncoder(model_path, local_files_only=True, **kwargs)
    except Exception:
        return CrossEncoder(model_path, **kwargs)


def run_rerank_batch(model, pairs, batch_size=8, warmup=1, rounds=3):
    """对多组 (query, passage) 做重排，测平均耗时"""
    for _ in range(warmup):
        model.predict(pairs[: min(batch_size, len(pairs))], batch_size=batch_size)
    if hasattr(torch, "cuda") and torch.cuda.is_available() and "cuda" in str(model.device):
        torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        model.predict(pairs, batch_size=batch_size)
        if hasattr(torch, "cuda") and torch.cuda.is_available() and "cuda" in str(model.device):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    device_cpu = "cpu"
    device_gpu = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else None)

    model_path = get_rerank_model_path()
    print(f"重排模型: {model_path}")
    print(f"  CPU: {device_cpu}")
    print(f"  GPU/加速: {device_gpu or '不可用'}")
    print()

    # 构造与业务类似的 (query, passage) 对
    query = "桥梁支座病害类型有哪些？如何养护？"
    passages = [
        "桥梁支座分为板式橡胶支座、盆式支座等，常见病害有开裂、脱空、剪切变形。",
        "支座养护应定期检查位移与变形，及时更换损坏支座。",
        "桥面铺装病害包括裂缝、坑槽、车辙等。",
        "伸缩缝应保持清洁、无杂物，橡胶条老化需更换。",
        "墩台裂缝可采用表面封闭或压力注浆处理。",
    ] * 4  # 20 对，模拟约 10 条候选
    pairs = [[query, p] for p in passages]

    # CPU
    print("加载 CPU 模型...")
    model_cpu = load_model(device_cpu)
    t_cpu = run_rerank_batch(model_cpu, pairs, batch_size=8, warmup=1, rounds=3)
    print(f"  CPU 平均耗时: {t_cpu:.3f}s ({len(pairs)} 对)")
    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # GPU / MPS
    if device_gpu:
        print("加载 GPU/MPS 模型...")
        model_gpu = load_model(device_gpu)
        t_gpu = run_rerank_batch(model_gpu, pairs, batch_size=8, warmup=1, rounds=3)
        print(f"  {device_gpu.upper()} 平均耗时: {t_gpu:.3f}s ({len(pairs)} 对)")
        speedup = t_cpu / t_gpu if t_gpu > 0 else 0
        print(f"  加速比: {speedup:.2f}x")
        del model_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("未检测到 GPU/MPS，仅测试 CPU。")

    print("\n完成。")


if __name__ == "__main__":
    main()
