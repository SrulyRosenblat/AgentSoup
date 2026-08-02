"""CLI: `agentsoup run steps.py` and `agentsoup status [RUN_ID]`."""
import argparse
import json

from .pipeline import Pipeline, list_runs, load_run, new_run_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentsoup")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a pipeline file")
    run_p.add_argument("file")
    run_p.add_argument("--input")
    run_p.add_argument("--run-id")
    run_p.add_argument("--state-dir", default=".agentsoup/runs")
    run_p.add_argument("--webhook")

    status_p = sub.add_parser("status", help="show run status from the state dir")
    status_p.add_argument("run_id", nargs="?")
    status_p.add_argument("--state-dir", default=".agentsoup/runs")
    status_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        pipeline = Pipeline.from_file(args.file)
        run_id = args.run_id or new_run_id(args.file)
        print(f"run {run_id} ({len(pipeline.steps)} steps)")

        def on_event(event, step_state):
            if event.startswith("step_"):
                print(f"  [{step_state['index'] + 1}/{len(pipeline.steps)}] "
                      f"{step_state['name']}: {step_state['status']}")

        try:
            result = pipeline.run(
                input=args.input, run_id=run_id, state_dir=args.state_dir,
                webhook_url=args.webhook, on_event=on_event,
            )
        except Exception as e:
            print(f"failed: {type(e).__name__}: {e}")
            return 1
        print(f"done. output: {result.output}")
        print(f"state: {result.state_path}")
        return 0

    if args.run_id:
        run = load_run(f"{args.state_dir}/{args.run_id}.json")
        if args.json:
            print(json.dumps(run, indent=2))
            return 0
        print(f"{run['run_id']}: {run['status']} (updated {run['updated_at']})")
        for s in run["steps"]:
            duration = f" {s['duration_s']}s" if s["duration_s"] is not None else ""
            error = f" — {s['error']}" if s["error"] else ""
            print(f"  {s['index'] + 1}. {s['name']}: {s['status']}{duration}{error}")
        return 0

    runs = list_runs(args.state_dir)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print(f"no runs in {args.state_dir}")
        return 0
    for run in runs:
        done = sum(1 for s in run["steps"] if s["status"] == "done")
        print(f"{run['run_id']}  {run['status']}  {done}/{len(run['steps'])} steps  {run['updated_at']}")
    return 0
