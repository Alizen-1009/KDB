import threading
import time

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = "/Users/alizen/models/Qwen3-0.6B"
device = "mps"

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
mdl.to(device)
mdl.eval()
app = FastAPI()


@app.post("/v1/chat/completions")
def chat(r: dict):
    p = tok.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=True
    )
    x = tok(p, return_tensors="pt").to(device)
    y = mdl.generate(**x, max_new_tokens=r.get("max_tokens", 64))
    t = tok.decode(y[0][x["input_ids"].shape[-1] :], skip_special_tokens=True)
    return t


threading.Thread(
    target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error"),
    daemon=True,
).start()
time.sleep(1)
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
rsp = client.chat.completions.create(
    model=model_path,
    messages=[{"role": "user", "content": 'Type "I love Qwen3.6" backwards'}],
    max_tokens=100,
)
print(rsp)
