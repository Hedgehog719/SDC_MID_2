"""Tree-trunk boxes with LocateAnything-3B (nvidia, open weights, NVIDIA non-commercial research license) for the frames
of the two public passes.  Inference follows the model card's worker (trust_remote_code, bf16), with the card's sampling
settings and a fixed seed per frame (greedy decoding without the repetition penalty loops into hundreds of duplicate boxes).  Saved: <target>/result/la_trunks.npz (frame, x1, y1, x2, y2 in full-res pixels).
usage (env with transformers 4.57.1): python process/la_trunks.py Track2 [--frames 990:1220,2870:3110] [--query "tree trunk"]"""
import argparse, os, re, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("--frames", default="990:1220,2870:3110"); ap.add_argument("--query", default="tree trunk")
ap.add_argument("--mode", default="hybrid"); ap.add_argument("--model", default="nvidia/LocateAnything-3B")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sorted(os.listdir(os.path.join(ROOT, a.target, "data", "images")))
frames = [f for r in a.frames.split(",") for f in range(*(int(x) for x in r.split(":")))]
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
prompt = f"Locate all the instances that match the following description: {a.query}."
rows = []
t0 = time.time()
for n, f in enumerate(frames):
    img = Image.open(os.path.join(ROOT, a.target, "data", "images", names[f])).convert("RGB")
    W, H = img.size
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(msgs)
    inp = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    torch.manual_seed(f)                                     # model card settings (sampling) with a fixed seed per frame
    with torch.no_grad():
        out = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16), input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                             image_grid_hws=inp.get("image_grid_hws", None), tokenizer=tok, max_new_tokens=4096, use_cache=True,
                             generation_mode=a.mode, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, verbose=False)
    ans = out[0] if isinstance(out, tuple) else out
    if isinstance(ans, list):
        ans = ans[0]
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        rows.append((f, x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H))
    if n % 20 == 0:
        print(f"{n}/{len(frames)} frames, {len(rows)} boxes  ({time.time() - t0:.0f}s)", flush=True)
r = np.array(rows, dtype=np.float64).reshape(-1, 5)
np.savez(os.path.join(ROOT, a.target, "result", "la_trunks.npz"), frame=r[:, 0].astype(int), x1=r[:, 1], y1=r[:, 2], x2=r[:, 3], y2=r[:, 4])
print(f"saved {len(r)} boxes from {len(frames)} frames  ({time.time() - t0:.0f}s)")
