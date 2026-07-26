"""Step 0：环境自检 —— 依赖是否装齐、GPU 是否可用。可直接运行。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    print("Python", sys.version.split()[0])
    ok = True
    for mod in ["torch", "open_clip", "faiss", "numpy", "sklearn", "yaml", "pandas", "matplotlib",
                "tqdm", "PIL", "huggingface_hub"]:
        try:
            m = __import__(mod)
            print("  [ok]      %-13s %s" % (mod, getattr(m, "__version__", "")))
        except Exception as e:
            ok = False
            print("  [MISSING] %-13s %s" % (mod, e))
    try:
        import torch
        av = torch.cuda.is_available()
        print("  cuda available:", av, "| device:", torch.cuda.get_device_name(0) if av else "-")
    except Exception:
        pass
    print("OK" if ok else "缺依赖，见上；按 README / 实现方案 Step 0 安装")


if __name__ == "__main__":
    main()
