import os

key = os.getenv("GEMINI_API_KEY")

if key is None:
    print("❌ Still not found")
else:
    print("✅ Key found:", key[:8], "...")