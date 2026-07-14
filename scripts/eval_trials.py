#!/usr/bin/env python3
"""Live-trial eval logger for real-robot policy runs — the missing "what's the success rate?" tool.

Runs alongside the normal control GUI: you drive the robot exactly as today (press 启动 /
start-auto-inference), and score each attempt here with a single keystroke. This tool never
touches the robot, the SDK, or the policy server — it only records outcomes and timing — so it is
task-agnostic (use the same tool for a second task) and safe to run on any machine.

Per trial it records: outcome (success / a failure mode), duration, and a wall-clock timestamp.
It writes one JSON object per line to infer_logs/eval/<session>.jsonl (append-only, so a session
resumes cleanly after a crash), and prints a live summary — success rate with a Wilson 95%
confidence interval, mean±std completion time, and a failure-mode histogram — the exact numbers a
resume bullet / report needs.

Operator flow, one trial at a time:
    SPACE   mark trial START   (press it the moment you hit 启动 / the robot begins the attempt)
    1..N    score the OUTCOME  (1 = success; 2.. = the failure modes below) -> ends the trial
    u       undo the last scored trial (typo insurance)
    n       add a free-text note to the last trial
    s       print the running summary
    q       quit (prints the final summary)

Usage:
    .venv/bin/python scripts/eval_trials.py --task flip_place
    .venv/bin/python scripts/eval_trials.py --task pour --session pour_v2
    .venv/bin/python scripts/eval_trials.py --categories "success,grasp-miss,dropped,misplaced,estop,other"
    .venv/bin/python scripts/eval_trials.py --summary infer_logs/eval/flip_place.jsonl   # re-print stats, score nothing

The outcome categories are configurable; index 1 must always be the success category. Defaults suit
a grasp -> lift -> flip -> place task; pass --categories for a different task.
"""
import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import sys
import time
from collections import deque

# Default outcome taxonomy for the grasp/flip/place task. Index 0 ("success") is special: everything
# else is a distinct FAILURE MODE, so the histogram tells you WHERE the policy loses attempts (the
# most useful thing for deciding what to improve next). Override with --categories for another task.
DEFAULT_CATEGORIES = [
    "success",       # 1: object ended up placed as intended
    "grasp-miss",    # 2: never got a good grasp / fingers closed on nothing
    "dropped",       # 3: grasped then lost it mid lift/flip
    "misplaced",     # 4: placed but wrong pose / off target / knocked over
    "collision",     # 5: self-collision / sim-validation reject / hit the scene
    "estop",         # 6: operator or firmware E-stop (aborted, not a policy failure)
    "other",         # 7: anything else (add a note with 'n')
]

EVAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "infer_logs" / "eval"


# Log-line markers the GUI emits (real_world/grasp_recovery.py, flip_place.py, inference_controller.py).
# The watcher turns each appended line into an event; the tracker turns events into trial boundaries.
#   START:    a run begins (homes to the start pose) -> the FIRST attempt starts.
#   RETREAT:  grasp-recovery opened the gripper and moved home after a missed grasp = one retry.
#   COMPLETE: the flip-place macro finished (object placed, arm withdrawn) -> attempt done AND the
#             next attempt begins (continuous auto has no per-attempt homing between placements).
LOG_MARKERS = [
    (b"[auto] homing to start pose", "start"),
    (b"[recovery] missed grasp",     "retreat"),
    (b"[flip-place] macro complete", "complete"),
]


class LogWatcher:
    """Tails the GUI log and streams (kind, ts) events for the markers above. poll() reads only the
    bytes appended since the last call and classifies each COMPLETE line, byte-offset based so it is
    robust to UTF-8 and to the file not existing yet (it starts watching once it appears). `ts` is
    the monotonic time the line was observed — accurate to the poll interval, which is plenty for
    multi-second attempts. available() is False when watching is disabled (path is None)."""

    def __init__(self, path):
        self.path = pathlib.Path(path) if path else None
        self._pos = None          # byte offset read up to; None until the file first appears
        self._buf = b""           # partial trailing line carried between polls

    def available(self) -> bool:
        return self.path is not None

    def _ensure_open(self):
        """Lazily latch onto the file the first time it exists, seeking to its END so only NEW lines
        (this eval session's trials) produce events — never the backlog from a previous run."""
        if self.path is None or self._pos is not None or not self.path.exists():
            return
        self._pos = self.path.stat().st_size

    def poll(self):
        """Return the list of (kind, monotonic_ts) events appended since the last poll (possibly [])."""
        self._ensure_open()
        if self._pos is None:
            return []
        try:
            size = self.path.stat().st_size
            if size < self._pos:                 # file truncated/rotated -> restart from its start
                self._pos, self._buf = 0, b""
            with self.path.open("rb") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return []
        data = self._buf + chunk
        lines = data.split(b"\n")
        self._buf = lines.pop()                  # last element is an incomplete line (no trailing \n)
        now = time.monotonic()
        events = []
        for line in lines:
            for needle, kind in LOG_MARKERS:
                if needle in line:
                    events.append((kind, now))
                    break
        return events


class TrialTracker:
    """Turns the watcher's ordered events into trials. An `active` attempt accumulates retreats until
    a COMPLETE finalizes it into the `pending` queue (awaiting the operator's outcome digit) and
    immediately opens the next attempt. Scoring pops the oldest pending; if none is pending it scores
    the in-progress attempt (manual early call) or, with no attempt at all, an untimed manual score."""

    def __init__(self):
        self.active = None                       # {"start_ts", "retreats", "auto"} or None
        self.pending = deque()                   # finalized-but-unscored {"duration_s","grasp_misses","grasp_src"}

    def start(self, ts, auto):
        self.active = {"start_ts": ts, "retreats": 0, "auto": auto}

    def retreat(self):
        if self.active is not None:
            self.active["retreats"] += 1
            return self.active["retreats"]
        return None

    def _finalize(self, ts):
        a = self.active
        return {"duration_s": round(ts - a["start_ts"], 2),
                "grasp_misses": a["retreats"],
                "grasp_src": "log" if a["auto"] else "manual"}

    def complete(self, ts):
        """COMPLETE event: finalize the active attempt into pending, then open the next one."""
        fin = self._finalize(ts) if self.active is not None else None
        if fin is not None:
            self.pending.append(fin)
        self.start(ts, auto=True)                # the placement also begins the next attempt
        return fin

    def score(self, ts):
        """Return the timing/retreat dict to attach the outcome to (never None; falls back to untimed)."""
        if self.pending:
            return self.pending.popleft()
        if self.active is not None:
            fin = self._finalize(ts)
            self.active = None
            return fin
        return {"duration_s": None, "grasp_misses": None, "grasp_src": None}


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial success rate — the right CI for small n (a handful of
    trials), unlike the naive normal interval which underflows/overshoots [0,1]. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fmt_duration(secs: float) -> str:
    return f"{secs:5.1f}s"


def load_trials(path: pathlib.Path):
    """Read an append-only session file back into a list of trial dicts (skips 'undo' tombstones)."""
    if not path.exists():
        return []
    trials = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_op") == "undo":
                if trials:
                    trials.pop()
                continue
            if rec.get("_op") == "note":
                if trials:
                    trials[-1].setdefault("note", "")
                    trials[-1]["note"] = rec.get("note", "")
                continue
            trials.append(rec)
    return trials


def summarize(trials, categories):
    """Print success rate (+Wilson 95% CI), timing, and a failure-mode histogram for `trials`."""
    n = len(trials)
    print("\n" + "=" * 58)
    if n == 0:
        print("  no trials scored yet")
        print("=" * 58)
        return
    success_cat = categories[0]
    succ = sum(1 for t in trials if t["outcome"] == success_cat)
    lo, hi = wilson_interval(succ, n)
    print(f"  TRIALS: {n}    SUCCESS: {succ}/{n} = {100.0*succ/n:.1f}%"
          f"   (95% CI {100*lo:.0f}–{100*hi:.0f}%)")

    durs = [t["duration_s"] for t in trials if t.get("duration_s") is not None]
    if durs:
        mean = sum(durs) / len(durs)
        std = (sum((d - mean) ** 2 for d in durs) / len(durs)) ** 0.5
        succ_durs = [t["duration_s"] for t in trials
                     if t["outcome"] == success_cat and t.get("duration_s") is not None]
        line = f"  TIME:   mean {_fmt_duration(mean)} ± {std:4.1f}s over {len(durs)} timed"
        if succ_durs:
            line += f"   (success-only mean {_fmt_duration(sum(succ_durs)/len(succ_durs))})"
        print(line)

    # First-try pickup vs recovery. `grasp_misses` counts how many times the policy closed on
    # nothing and re-approached before the attempt ended (0 = clean first-try grasp). Comparing
    # first-try SUCCESS to total SUCCESS quantifies what the auto-recovery system is worth.
    tracked = [t for t in trials if t.get("grasp_misses") is not None]
    if tracked:
        first_try = sum(1 for t in tracked if t["grasp_misses"] == 0)
        mean_retries = sum(t["grasp_misses"] for t in tracked) / len(tracked)
        succ_tracked = [t for t in tracked if t["outcome"] == success_cat]
        succ_first = sum(1 for t in succ_tracked if t["grasp_misses"] == 0)
        lo_ft, hi_ft = wilson_interval(first_try, len(tracked))
        print(f"  GRASP:  first-try (0 retries): {first_try}/{len(tracked)} = "
              f"{100.0*first_try/len(tracked):.1f}%   (95% CI {100*lo_ft:.0f}–{100*hi_ft:.0f}%)")
        print(f"          mean {mean_retries:.2f} retries/trial", end="")
        if succ_tracked:
            print(f"   ·   of {len(succ_tracked)} successes, {succ_first} were first-try, "
                  f"{len(succ_tracked)-succ_first} needed recovery", flush=True)
        else:
            print(flush=True)

    print("  OUTCOMES:")
    counts = {c: 0 for c in categories}
    for t in trials:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    width = max(len(c) for c in categories)
    for c in categories:
        k = counts.get(c, 0)
        bar = "█" * k
        tag = "  ✓" if c == success_cat else ("  ✗" if k else "   ")
        print(f"    {tag} {c:<{width}}  {k:>3}  {bar}")
    print("=" * 58)


# --------------------------------------------------------------------------- #
#  Keyboard input that can also time out, so the main loop can poll the log     #
#  between keystrokes. cbreak (not raw) keeps Ctrl-C working; non-TTY stdin     #
#  (piped/redirected) still works so the tool stays scriptable and testable.    #
# --------------------------------------------------------------------------- #
import select                                                   # noqa: E402


class KeyReader:
    """Context manager yielding get(timeout) -> one char, "" on EOF, or None if `timeout` elapsed
    with no key. On a TTY it puts the terminal in cbreak mode for the session (restored on exit) and
    uses select() for the timeout; on a pipe it selects/reads the fd the same way."""

    def __init__(self):
        self.tty = sys.stdin.isatty()
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        if self.tty:
            import termios, tty
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        self._restore()

    def _restore(self):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None

    def get(self, timeout):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None                                         # timed out -> caller polls the log
        return sys.stdin.read(1)                                # "" on EOF

    def cooked_line(self, prompt):
        """Temporarily restore normal line editing to read a full line (for the note prompt)."""
        saved, self._old = self._old, None
        if saved is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, saved)
        try:
            return input(prompt).strip()
        except EOFError:
            return ""
        finally:
            if saved is not None:
                import tty
                tty.setcbreak(self._fd)
                self._old = saved


def _print_help(categories, auto):
    start = "log auto-starts/completes trials" if auto else "SPACE=start trial"
    retreat = "" if auto else "g=grasp miss (re-approach)   "
    print(f"\n  keys:  {start}   {retreat}" +
          "  ".join(f"{i+1}={c}" for i, c in enumerate(categories)))
    print("         u=undo   n=note   s=summary   q=quit")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="task",
                    help="task label (also the default session filename)")
    ap.add_argument("--session", default=None,
                    help="session name -> infer_logs/eval/<session>.jsonl (default: --task)")
    ap.add_argument("--categories", default=None,
                    help="comma-separated outcome list; index 1 MUST be the success category")
    ap.add_argument("--summary", default=None, metavar="FILE",
                    help="just print the summary for an existing session file and exit")
    ap.add_argument("--log", default=str(EVAL_DIR.parent / "gui.log"), metavar="FILE",
                    help="GUI log to auto-count '[recovery] missed grasp' retreats "
                         "(default infer_logs/gui.log; 'none' to disable and score retries by hand)")
    args = ap.parse_args()

    categories = ([c.strip() for c in args.categories.split(",")]
                  if args.categories else list(DEFAULT_CATEGORIES))

    if args.summary:
        path = pathlib.Path(args.summary)
        trials = load_trials(path)
        # Re-derive categories from the file if the caller didn't pass any, so the histogram is complete.
        if not args.categories:
            seen = list(dict.fromkeys(t["outcome"] for t in trials))
            for c in seen:
                if c not in categories:
                    categories.append(c)
        summarize(trials, categories)
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    session = args.session or args.task
    path = EVAL_DIR / f"{session}.jsonl"
    trials = load_trials(path)

    watcher = LogWatcher(None if args.log.lower() == "none" else args.log)
    tracker = TrialTracker()
    auto = watcher.available()

    print(f"eval session: {path}")
    print(f"task:         {args.task}")
    if not auto:
        print("trials:       manual — SPACE to start, 'g' per missed grasp, digit to score")
    else:
        exists = watcher.path.exists()
        print(f"trials:       AUTO from {watcher.path}"
              f"{'' if exists else ' (once the GUI creates it)'} — "
              f"starts/retreats/completes read from the log; you only press the outcome digit")
    if trials:
        print(f"resuming — {len(trials)} trial(s) already scored")
    _print_help(categories, auto)
    if trials:
        summarize(trials, categories)

    logf = path.open("a")

    def append(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    def record(idx, base):
        """Attach outcome `idx` to a finalized timing/retreat `base` dict, persist, and echo."""
        rec = {
            "task": args.task,
            "trial": len(trials) + 1,
            "outcome": categories[idx],
            "duration_s": base["duration_s"],
            "grasp_misses": base["grasp_misses"],
            "grasp_src": base["grasp_src"],
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        append(rec)
        trials.append(rec)
        dtxt = _fmt_duration(rec["duration_s"]) if rec["duration_s"] is not None else "  --  "
        gtxt = f" {rec['grasp_misses']}r" if rec["grasp_misses"] is not None else ""
        succ = sum(1 for t in trials if t["outcome"] == categories[0])
        print(f"  {'✓' if idx == 0 else '✗'} trial {rec['trial']}: {rec['outcome']:<12} "
              f"{dtxt}{gtxt}   running: {succ}/{len(trials)} = {100.0*succ/len(trials):.0f}%",
              flush=True)

    print("\n> ready. " + ("waiting for the GUI to start a run…" if auto
                           else "SPACE to start a trial."), flush=True)

    POLL_S = 0.2   # how often to check the log for new events while waiting on a key
    with KeyReader() as reader:
        while True:
            # 1) Drain any new log events into the tracker (auto trial boundaries + retreats).
            for kind, ts in watcher.poll():
                if kind == "start":
                    tracker.start(ts, auto=True)     # run homed -> fresh attempt (drops any phantom)
                    print(f"  ▶ trial {len(trials)+1} started (run homing)", flush=True)
                elif kind == "retreat":
                    n = tracker.retreat()
                    if n is not None:
                        print(f"  ⟳ retreat #{n} (missed grasp -> re-approach)", flush=True)
                elif kind == "complete":
                    fin = tracker.complete(ts)
                    if fin is not None:
                        rtxt = f", {fin['grasp_misses']} retreat(s)" if fin["grasp_misses"] else ""
                        depth = len(tracker.pending)
                        print(f"  ✅ attempt complete ({_fmt_duration(fin['duration_s'])}{rtxt}) — "
                              f"press the outcome digit (1={categories[0]}, 2..=failure)"
                              f"{f'  [{depth} awaiting score]' if depth > 1 else ''}", flush=True)

            # 2) Wait briefly for a key; None => timed out, loop back and poll the log again.
            ch = reader.get(POLL_S)
            if ch is None:
                continue
            if ch in ("\x03", "\x04"):               # Ctrl-C / Ctrl-D
                ch = "q"

            if ch == " ":
                tracker.start(time.monotonic(), auto=False)
                print(f"  ▶ trial {len(trials)+1} started (manual)", flush=True)

            elif ch == "g":
                n = tracker.retreat()
                if n is None:
                    print("  (no trial in progress — start one first)", flush=True)
                else:
                    print(f"  ⟳ grasp miss #{n} (re-approach)", flush=True)

            elif ch.isdigit() and ch != "0":
                idx = int(ch) - 1
                if idx >= len(categories):
                    print(f"  (no category {ch}; there are {len(categories)})", flush=True)
                    continue
                record(idx, tracker.score(time.monotonic()))

            elif ch == "u":
                if not trials:
                    print("  (nothing to undo)", flush=True)
                    continue
                popped = trials.pop()
                append({"_op": "undo", "ts": _dt.datetime.now().isoformat(timespec="seconds")})
                print(f"  ↩ undid trial {popped['trial']} ({popped['outcome']})", flush=True)

            elif ch == "n":
                if not trials:
                    print("  (score a trial first, then add a note)", flush=True)
                    continue
                note = reader.cooked_line("  note> ")
                if note:
                    trials[-1]["note"] = note
                    append({"_op": "note", "note": note,
                            "ts": _dt.datetime.now().isoformat(timespec="seconds")})
                    print(f"  📝 note on trial {trials[-1]['trial']}", flush=True)

            elif ch == "s":
                summarize(trials, categories)

            elif ch in ("q", ""):
                break

            elif ch == "?":
                _print_help(categories, auto)

    logf.close()
    summarize(trials, categories)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
