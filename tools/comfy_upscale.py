"""comfy_upscale.py -- batch-upscale images through a running ComfyUI instance.

Copies inputs to D:\\input\\<job>\\, queues LoadImage -> UpscaleModelLoader ->
ImageUpscaleWithModel -> SaveImage per image, waits for completion, and reports
the output files (written to D:\\output\\<job>\\ with an UPS_ prefix so runs
from the two instances can't collide with anything else).

  python comfy_upscale.py "glob_or_folder" --port 8188 --model 4xNomosWebPhoto_RealPLKSR.safetensors --job UPS_pilot
"""
import argparse, glob, json, os, shutil, sys, time, urllib.error, urllib.request

COMFY_INPUT = os.environ.get("COMFY_INPUT", r"D:\input")
COMFY_OUTPUT = os.environ.get("COMFY_OUTPUT", r"D:\output")


def post(port, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def history(port, pid):
    """Transient server errors must not kill the poll loop -- report as 'not yet'."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as r:
            return json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def graph(image_rel, model, out_prefix):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_rel}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": out_prefix}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--model", default="4xNomosWebPhoto_RealPLKSR.safetensors")
    ap.add_argument("--job", default="UPS_job")
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            files += glob.glob(os.path.join(item, "*.png")) + glob.glob(os.path.join(item, "*.jpg"))
        else:
            files += glob.glob(item)
    files = sorted(set(files))
    if not files:
        sys.exit("no inputs")

    stage = os.path.join(COMFY_INPUT, args.job)
    os.makedirs(stage, exist_ok=True)
    tag = args.model.split(".")[0].split("_")[0]

    pending = {}
    for p in files:
        base = os.path.basename(p)
        shutil.copy2(p, os.path.join(stage, base))
        prefix = f"{args.job}/UPS_{tag}_{os.path.splitext(base)[0]}"
        try:
            r = post(args.port, {"prompt": graph(f"{args.job}/{base}", args.model, prefix)})
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"QUEUE FAILED {base}: {e}")
            continue
        pending[r["prompt_id"]] = base
    print(f"queued {len(pending)} on :{args.port} with {args.model}")

    done, t0 = {}, time.time()
    while pending and time.time() - t0 < 1800:
        time.sleep(2)
        for pid in list(pending):
            h = history(args.port, pid)
            if pid not in h:
                continue
            st = h[pid].get("status", {})
            if st.get("completed"):
                outs = [o["filename"] for node in h[pid].get("outputs", {}).values() for o in node.get("images", [])]
                done[pending.pop(pid)] = outs
            elif st.get("status_str") == "error":
                print(f"ERROR {pending.pop(pid)}")
    for src, outs in sorted(done.items()):
        print(f"  {src} -> {outs}")
    if pending:
        print(f"TIMED OUT waiting on {len(pending)}: {list(pending.values())}")
    print(f"outputs in {os.path.join(COMFY_OUTPUT, args.job)}")


if __name__ == "__main__":
    main()
