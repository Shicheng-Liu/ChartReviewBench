"""Exercise only synthetic local inputs; never call a model or read credentials."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".smoke"


def run(args):
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               CRB_PYTHON=sys.executable)
    result = subprocess.run([sys.executable, "-B", *map(str, args)], cwd=ROOT,
                            env=env, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
    return result.stdout


def main():
    WORK.mkdir(exist_ok=True)
    inst = ROOT / "examples/title"
    inst.mkdir(parents=True, exist_ok=True)
    template = ("import matplotlib.pyplot as plt\n"
                "plt.plot([0,1,2],[1,2,3], label='Observations')\n"
                "plt.title({title!r})\nplt.legend()\nplt.savefig('output.png')\n")
    (inst / "ground_truth.py").write_text(template.format(title="Observations"))
    (inst / "flawed.py").write_text(template.format(title="Wrong title"))
    (inst / "table.csv").write_text("x,y\n0,1\n1,2\n2,3\n")
    (inst / "summary.txt").write_text("A line chart titled Observations shows y=1,2,3 for x=0,1,2.\n")
    common = ["scoring/score_submission.py", "--instance", inst]
    checks = {}
    for name, submission, options in (
        ("repaired", "ground_truth.py", ["--category", "wrong-title"]),
        ("unrepaired", "flawed.py", ["--category", "wrong-title"]),
        ("clean", "ground_truth.py", ["--clean"]),
    ):
        result = json.loads(run(common + ["--submission", inst / submission] + options))
        assert result["executability"] == 1
        expected = {"repaired": 1, "unrepaired": 0, "clean": None}[name]
        assert result["recovery_binary"]["strict"] == expected
        assert result["preservation_binary"]["strict"] == 1
        checks[name] = {"executability": 1, "recovery_binary": expected,
                        "preservation_binary": 1}
    if sys.platform == "darwin":
        result = json.loads(run(common + ["--submission", inst / "ground_truth.py",
                                         "--category", "wrong-title", "--isolate"]))
        assert result["recovery_binary"]["strict"] == 1
        checks["isolated_scoring"] = "passed"

        sample = WORK / "samples/example001"
        gt, flawed = sample / "gt", sample / "perturbed_llm/wrong-title"
        gt.mkdir(parents=True, exist_ok=True)
        flawed.mkdir(parents=True, exist_ok=True)
        for name in ("table.csv", "summary.txt"):
            shutil.copyfile(inst / name, gt / name)
        shutil.copyfile(inst / "ground_truth.py", gt / "code.py")
        shutil.copyfile(inst / "flawed.py", flawed / "code.py")
        from scoring.chartrepair_eval.capture import capture_path
        for directory in (gt, flawed):
            captured = capture_path(directory / "code.py")
            assert captured["status"] == "ok"
            (directory / "render.png").write_bytes(captured["png"])
        (flawed / "perturbation.json").write_text(json.dumps({
            "category": "wrong-title", "description": "Title changed", "sample_folder": "example001"}))
        dataset = WORK / "dataset.json"
        dataset.write_text(json.dumps({"name": "synthetic-smoke", "total": 1,
            "items": [{"instance": "example001", "category": "wrong-title"}]}))
        run(["-m", "src.eval.run_benchmark", "--offline", "--dataset", dataset,
             "--samples", WORK / "samples", "--tasks", WORK / "tasks",
             "--out", WORK / "run", "--track", "B", "--max-steps", "3",
             "--wall-time-limit", "60", "--workers", "1", "--force"])
        episodes = list((WORK / "run/episodes").glob("*.json"))
        assert len(episodes) == 1
        episode = json.loads(episodes[0].read_text())
        assert episode["status"] == "ok", episode.get("error")
        assert episode["metrics"]["recovery_binary"]["strict"] == 0
        checks["offline_agent"] = "passed: unchanged flawed chart has zero recovery"
    else:
        checks["isolated_scoring"] = checks["offline_agent"] = "not run: macOS required"
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
