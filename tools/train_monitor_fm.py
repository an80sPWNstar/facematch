"""train_monitor_fm.py -- grade LoRA training samples through the FaceMatch app.

Watches an ai-toolkit samples dir, scores each complete 5-image sample set
against a reference bank built from the training dataset (via the FaceMatch
Gradio API on :7861), and stops the training job through the ai-toolkit UI
API when likeness peaks.

  python train_monitor_fm.py --samples DIR --dataset DIR --csv out.csv \
      --job-id UUID --label mylora

Needs the FaceMatch app running (launch_facematch.bat). If the app goes away
mid-run it reconnects and rebuilds the bank; if the monitor itself is
restarted it skips steps already in the CSV and reloads their means so the
stop rules keep their history.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import urllib.request

from gradio_client import Client, handle_file

FACEMATCH_URL = "http://127.0.0.1:7861/"
UI_API = "http://localhost:8675/api"
POLL_S = 120
SETTLE_S = 90          # newest file of a set must be this old before scoring
SET_SIZE = 5
SCORE_THRESHOLD = 0.35  # only affects box color in the app, not the scores
OVERBAKE_MIN_SETS = 10
OVERBAKE_DROP = 0.02
PLATEAU_MIN_SETS = 16
PLATEAU_BEST_AGE = 12  # require a well-stale peak before a plateau stop
PLATEAU_EPS = 0.005
RETRY_S = 30

SAMPLE_RE = re.compile(r"_(\d{9})_(\d)\.jpg$", re.IGNORECASE)
CSV_HEADER = ["step", "mean", "s0", "s1", "s2", "s3", "s4", "time"]


def log(msg):
    print(msg, flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def connect_and_build_bank(dataset_dir):
    """Connect to FaceMatch and build the session bank. Retries forever."""
    refs = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    if not refs:
        sys.exit(f"FATAL: no .png images in dataset dir {dataset_dir}")
    while True:
        try:
            client = Client(FACEMATCH_URL, verbose=False)
            info = client.predict(files=[handle_file(p) for p in refs],
                                  api_name="/build_bank")
            log(f"bank: {info}")
            if "Bank built" not in str(info):
                raise RuntimeError(f"bank build did not succeed: {info}")
            return client
        except Exception as e:
            log(f"FaceMatch unreachable or bank build failed ({e!r}); "
                f"retrying in {RETRY_S}s (is launch_facematch.bat running?)")
            time.sleep(RETRY_S)


def scan_sample_sets(samples_dir):
    """step -> {img_index: path} for files matching *_NNNNNNNNN_i.jpg."""
    sets = {}
    for p in glob.glob(os.path.join(samples_dir, "*.jpg")):
        m = SAMPLE_RE.search(os.path.basename(p))
        if not m:
            continue
        step, idx = int(m.group(1)), int(m.group(2))
        if idx >= SET_SIZE:
            continue
        sets.setdefault(step, {})[idx] = p
    return sets


def set_is_ready(members):
    if len(members) < SET_SIZE:
        return False
    try:
        newest = max(os.path.getmtime(p) for p in members.values())
    except OSError:
        return False
    return time.time() - newest > SETTLE_S


def rescore_noface(client, path):
    """RetinaFace at det_size 640 misses faces that fill the frame, so a tight
    close-up reads NOFACE (seen live at step 1500). Downscale the image onto a
    same-size canvas and rescore — 50%, then 25%."""
    import cv2
    import numpy as np
    import tempfile
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return -1.0
    H, W = img.shape[:2]
    for scale in (0.5, 0.25):
        small = cv2.resize(img, (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, W, 3), np.uint8)
        y, x = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
        fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="fm_retry_")
        os.close(fd)
        cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(tmp)
        try:
            result = client.predict(files=[handle_file(tmp)],
                                    threshold=SCORE_THRESHOLD,
                                    api_name="/score_candidates")
            rows = result[0].get("data") if isinstance(result[0], dict) else result[0]
            for row in rows or []:
                try:
                    return float(row[1])
                except (TypeError, ValueError):
                    pass
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return -1.0


def score_set(client, members):
    """Return sims s0..s4 (NOFACE/unreadable/missing -> -1) and the set mean."""
    paths = [members[i] for i in sorted(members)]
    result = client.predict(files=[handle_file(p) for p in paths],
                            threshold=SCORE_THRESHOLD,
                            api_name="/score_candidates")
    table = result[0]
    rows = table.get("data") if isinstance(table, dict) else table
    by_name = {}
    for row in rows or []:
        name, sim = row[0], row[1]
        try:
            by_name[name] = float(sim)
        except (TypeError, ValueError):
            by_name[name] = -1.0  # NOFACE
    sims = []
    for i in sorted(members):
        sims.append(by_name.get(os.path.basename(members[i]), -1.0))
    for i, s in enumerate(sims):
        if s < 0:
            sims[i] = rescore_noface(client, paths[i])
    valid = [s for s in sims if s >= 0]
    mean = sum(valid) / len(valid) if valid else -1.0
    return sims, mean


def load_done_from_csv(csv_path):
    """Steps already scored, and their (step, mean) history, from a prior run."""
    history = []
    if not os.path.isfile(csv_path):
        return history
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0] == "step":
                continue
            if any(v.startswith("-1.0") for v in row[2:7]):
                continue  # had a NOFACE — rescore this step after the retry fix
            try:
                history.append((int(row[0]), float(row[1])))
            except ValueError:
                continue  # STOP_* marker rows etc.
    history.sort()
    return history


def append_csv(csv_path, row):
    new = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def http_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def job_status(job_id):
    try:
        job = http_json(f"{UI_API}/jobs?id={job_id}")
        return (job or {}).get("status")
    except Exception as e:
        log(f"warning: could not read job status ({e!r})")
        return None


def stop_job(job_id):
    # GET /api/jobs/<id>/stop sets the stop flag; the trainer's own watcher
    # runs the graceful shutdown (never taskkill from here)
    try:
        job = http_json(f"{UI_API}/jobs/{job_id}/stop", timeout=30)
        log(f"stop requested via UI API; job status now: {(job or {}).get('status')}")
        return True
    except Exception as e:
        log(f"ERROR: stop request failed ({e!r}) — stop the job manually in the "
            f"ai-toolkit dashboard (http://localhost:8675)")
        return False


def check_stop_rules(history):
    """history: [(step, mean)] in step order. Returns (reason, best_step) or None."""
    means = [m for _, m in history]
    if not means:
        return None
    best = max(means)
    best_idx = means.index(best)
    best_step = history[best_idx][0]
    if len(means) >= OVERBAKE_MIN_SETS and all(m <= best - OVERBAKE_DROP
                                               for m in means[-3:]):
        return "OVERBAKE", best_step
    if (len(means) >= PLATEAU_MIN_SETS
            and len(means) - 1 - best_idx >= PLATEAU_BEST_AGE
            and all(m < best - PLATEAU_EPS for m in means[-PLATEAU_BEST_AGE:])):
        return "PLATEAU", best_step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="ai-toolkit samples dir")
    ap.add_argument("--dataset", required=True, help="bank source dir (*.png)")
    ap.add_argument("--csv", required=True, help="append scores here")
    ap.add_argument("--job-id", required=True, help="ai-toolkit job UUID")
    ap.add_argument("--label", required=True, help="name used in log lines")
    args = ap.parse_args()

    history = load_done_from_csv(args.csv)
    done = {s for s, _ in history}
    if done:
        log(f"[{args.label}] resuming: {len(done)} set(s) already in {args.csv}")

    client = connect_and_build_bank(args.dataset)

    while True:
        status = job_status(args.job_id)
        if status in ("error", "stopped", "completed", "failed"):
            log(f"[{args.label}] job status is '{status}' — monitor exiting")
            return

        sets = scan_sample_sets(args.samples)
        for step in sorted(sets):
            if step in done:
                continue
            members = sets[step]
            if not set_is_ready(members):
                continue
            while True:
                try:
                    sims, mean = score_set(client, members)
                    break
                except Exception as e:
                    log(f"[{args.label}] scoring failed ({e!r}); reconnecting "
                        f"and rebuilding bank")
                    client = connect_and_build_bank(args.dataset)
            done.add(step)
            history.append((step, mean))
            history.sort()
            append_csv(args.csv, [step, f"{mean:.4f}",
                                  *(f"{s:.4f}" for s in sims), now_iso()])
            log(f"[{args.label}] step {step}: mean {mean:.4f} "
                f"[{' '.join(f'{s:.3f}' for s in sims)}]")

            hit = check_stop_rules(history)
            if hit:
                reason, best_step = hit
                best = max(m for _, m in history)
                log(f"[{args.label}] {reason}: best mean {best:.4f} at step "
                    f"{best_step} — stopping job {args.job_id}")
                stop_job(args.job_id)
                append_csv(args.csv, [f"STOP_{reason}", f"{best:.4f}",
                                      f"best_step={best_step}",
                                      "", "", "", "", now_iso()])
                return

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
