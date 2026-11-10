"""Run independent deployment trials and report basket-entry success rate only."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from deployment_script.evaluate_basket_criteria import (
    RULE, RULE_VERSION, FINAL_STATUSES, summarize_basket_results, write_json_atomic,
)


def run_trials(args, *, runner=None):
    """The runner argument supports isolated subprocess integration checks."""
    launch_directory = Path.cwd()
    config = Path(args.config).resolve()
    policy_config = json.loads(config.read_text(encoding="utf-8"))
    simulator = Path(args.sim_repo).resolve()
    if not simulator.is_dir():
        raise FileNotFoundError(f"Simulator repository not found: {simulator}")
    output = Path(args.output or PROJECT / "runs" / "deployments" /
                  ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Choose an empty evaluation directory: {output}")
    # All trials use the same policy configuration even if the original is edited.
    snapshot = output / "policy_config.json"
    write_json_atomic(snapshot, policy_config)
    runner = Path(runner or PROJECT / "deployment_script" / "run_isaacsim.py")
    write_json_atomic(output / "evaluation_config.json", {
        "requested_trials": args.episodes, "config_source": str(config),
        "sim_repo": str(simulator), "seconds": args.seconds,
        "observation_hz": args.observation_hz, "action_hz": args.action_hz,
        "execute_steps": args.execute_steps, "camera": args.camera,
        "headless": args.headless, "criterion": RULE,
        "note": "Independent randomized resets; setup errors are reported separately. No failed task trial is retried."})
    try:
        for number in range(1, args.episodes + 1):
            episode = output / f"episode_{number:04d}"
            command = [sys.executable, "-u", str(runner), "--sim-repo", str(simulator),
                       "--config", str(snapshot), "--output", str(episode),
                       "--seconds", str(args.seconds), "--observation-hz", str(args.observation_hz),
                       "--action-hz", str(args.action_hz), "--execute-steps", str(args.execute_steps),
                       "--camera", args.camera]
            if args.headless:
                command.append("--headless")
            print(f"[evaluation] trial {number}/{args.episodes}: {episode}", flush=True)
            launch_error = None
            returncode = None
            with (output / f"console_{number:04d}.log").open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(command, cwd=launch_directory, stdout=log,
                                            stderr=subprocess.STDOUT)
                    returncode = result.returncode
                except OSError as exc:
                    launch_error = f"Could not launch trial: {exc}"
                    log.write(launch_error + "\n")
            report_path = episode / "summary.json"
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict):
                    raise ValueError("Summary must be a JSON object")
            except (OSError, ValueError) as exc:
                report = {"status": "failed", "success": None, "evaluation_started": False,
                          "error": launch_error or f"No readable child summary: {exc}"}
            report.setdefault("basket_evaluation", {"rule_version": RULE_VERSION, "criterion": RULE})
            # Native failures can bypass Python finally; don't silently lose an attempt.
            if report.get("status") not in FINAL_STATUSES:
                report.update(status="failed", error="Child exited before finalizing the trial")
                if report.get("evaluation_started"):
                    report["success"] = bool(report.get("success", False))
                else:
                    report["success"] = None
            report["process_returncode"] = returncode
            write_json_atomic(report_path, report)
            totals = summarize_basket_results(output)
            rate = totals["basket_success_rate_percent"]
            shown_rate = f"{rate:.1f}%" if rate is not None else "N/A"
            print(f"[evaluation] basket={totals['successes']}/{totals['total_trials']} "
                  f"({shown_rate}); setup_errors={totals['setup_errors']}", flush=True)
    finally:
        totals = summarize_basket_results(output)
        print(f"[evaluation] results: {output / 'basket_success_rate.json'}", flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", metavar="DIRECTORY", help="Recount existing scored summaries only")
    parser.add_argument("--sim-repo")
    parser.add_argument("--config", default="configs/deploy_isaacsim.json")
    parser.add_argument("--episodes", type=int, help="Number of trial attempts; required to run trials")
    parser.add_argument("--output")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--observation-hz", type=float, default=30)
    parser.add_argument("--action-hz", type=float, default=15)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--camera", default="front_camera")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--gui", dest="headless", action="store_false")
    # Match the renderer used by direct run_isaacsim.py deployments.
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    if args.summarize:
        directory = Path(args.summarize)
        if not directory.is_dir():
            parser.error(f"Evaluation directory not found: {directory}")
        totals = summarize_basket_results(directory)
        print(json.dumps(totals, ensure_ascii=False, indent=2))
        return
    if not args.sim_repo or args.episodes is None or args.episodes < 1:
        parser.error("Running trials requires --sim-repo and --episodes >= 1")
    if args.seconds <= 0 or args.execute_steps < 1:
        parser.error("Duration and execute-steps must be positive")
    for rate in (args.observation_hz, args.action_hz):
        ticks = round(120 / rate) if rate > 0 else 0
        if ticks < 1 or abs(ticks * rate - 120) > 1e-6:
            parser.error("Observation/action rates must divide 120 Hz exactly")
    totals = run_trials(args)
    if totals["setup_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
