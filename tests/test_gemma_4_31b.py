import os
import time

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

prompt = (
    "In two sentences, explain projectile motion to a curious 8th grader."
)

start = time.time()
first_token_at = None
token_count = 0

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a concise, friendly physics tutor.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    model="gemma-4-31b",
    stream=True,
    max_completion_tokens=2048,
    temperature=0.2,
    top_p=1,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    if delta and first_token_at is None:
        first_token_at = time.time()
    token_count += 1
    print(delta, end="", flush=True)

elapsed = time.time() - start
ttft = (first_token_at - start) if first_token_at else float("nan")
print("\n\n--- stats ---")
print(f"time to first token: {ttft:.3f}s")
print(f"total time:          {elapsed:.3f}s")
print(f"stream chunks:       {token_count}")
