"""End-to-end check of session isolation and context retention.

Run the server first, then: .venv\\Scripts\\python.exe test_context.py
"""
import json
import sys
import urllib.request

# Replies may contain emoji or dashes the Windows console's default cp1252
# codepage cannot encode, which would crash printing rather than the test.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8000"


def post(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(req)


def new_session():
    return json.load(post("/api/session"))["session_id"]


def ask(session_id, message):
    """Send a message and collect the streamed reply."""
    parts = []
    with post("/api/chat", {"message": message, "session_id": session_id}) as resp:
        event = None
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event == "token":
                parts.append(json.loads(line[5:].strip())["text"])
            elif line.startswith("data:") and event == "error":
                raise RuntimeError(json.loads(line[5:].strip())["detail"])
    return "".join(parts).strip()


def turn(session_id, question):
    print(f"\n\033[1mUser:\033[0m {question}")
    answer = ask(session_id, question)
    print(f"\033[1mAssistant:\033[0m {answer}")
    return answer


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION A - follow-up chain")
    print("=" * 70)
    a = new_session()
    turn(a, "What is RAG?")
    turn(a, "What are its main components?")
    turn(a, "Can you explain the second one in detail?")

    print("\n" + "=" * 70)
    print("SESSION A - topic change mid-session")
    print("=" * 70)
    turn(a, "Forget that. What's a good recipe for pancakes?")

    print("\n" + "=" * 70)
    print("SESSION B - fresh session, must NOT know about session A")
    print("=" * 70)
    b = new_session()
    turn(b, "What are its main components?")

    hist_a = json.load(urllib.request.urlopen(f"{BASE}/api/sessions/{a}"))["messages"]
    hist_b = json.load(urllib.request.urlopen(f"{BASE}/api/sessions/{b}"))["messages"]
    print(f"\n\nSession A holds {len(hist_a)} messages; session B holds {len(hist_b)}.")
