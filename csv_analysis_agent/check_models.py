import sys
import io
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

try:
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        if not m.id.startswith("text-embedding"):
            print(" -", m.id)
except Exception as e:
    print(f"Error: {e}")
