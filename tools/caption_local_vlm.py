"""caption_local_vlm.py -- caption training images via a local llama.cpp vision server.

Writes ai-toolkit style .txt captions beside each image: "<trigger>, <short scene line>".
Sequential (tesla has one slot). Skips images that already have a .txt. Run with
any python that has stdlib only -- uses urllib.

  python caption_local_vlm.py IMG_DIR --trigger my_subject [--url http://127.0.0.1:8080]
"""
import argparse, base64, glob, json, os, sys, time, urllib.request

PROMPT = ("Describe this photo in one line of at most 14 words: setting, lighting, "
          "clothing, expression, framing (close-up / upper body / full body). "
          "Do not name or identify the person. Output only the line, no quotes.")


def jpeg_payload(img_path, max_side=768):
    """Downscale + JPEG-encode so the request stays ~100-200KB. Full-res upscaled
    PNGs (~10MB base64) can crash a llama.cpp server."""
    import cv2, numpy as np
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    s = max_side / max(img.shape[:2])
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def wait_up(url, timeout_s=1200):
    """llama.cpp answers 503 while the model loads; wait until it serves."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/props", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(20)
    return False


def caption(url, img_path):
    b64 = jpeg_payload(img_path)
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 100,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)["choices"][0]["message"]["content"].strip()
    return " ".join(out.replace("\n", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imgdir")
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--url", default=os.environ.get("LOCAL_VLM_URL", "http://127.0.0.1:8080"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.imgdir, "*.png")) + glob.glob(os.path.join(args.imgdir, "*.jpg")))
    done = fail = skip = 0
    t0 = time.time()
    if not wait_up(args.url):
        sys.exit("FATAL: server did not come up within 20 minutes")
    for i, p in enumerate(files):
        txt = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(txt):
            skip += 1
            continue
        line = None
        for attempt in (1, 2):
            try:
                line = caption(args.url, p)
                break
            except Exception as e:
                print(f"[{i+1}/{len(files)}] attempt {attempt} failed {os.path.basename(p)}: {e}")
                if attempt == 1:
                    wait_up(args.url)  # server may be reloading; wait, then retry once
        if line is None:
            fail += 1
            continue
        # strip any stray thinking/markup and cap length
        line = line.split("</think>")[-1].strip()[:160]
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(f"{args.trigger}, {line}\n")
        done += 1
        print(f"[{i+1}/{len(files)}] {os.path.basename(p)}: {line}")
    print(f"\ndone {done}, skipped {skip}, failed {fail} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
