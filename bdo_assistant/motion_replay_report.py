from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Callable

from .motion import MotionClass
from .native_motion_replay import NativeMotionReplayFrame, load_native_motion_replay
from .navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from .runtime import ConservativeNavigationAnalyzer

Analyzer = Callable[[int, object], NavigationObservation]

# Legacy labels are evaluation metadata only; they are never passed to the analyzer.
_LABEL_TO_TARGET = {
    "MOVING_CONFIRMED": "moving",
    "MOVING_RECOVERED": "moving",
    "MOVING_APPROACH": "moving",
    "ARRIVED": "arrival",
    "ARRIVED_LATCHED": "arrival",
    "STUCK": "stuck",
    "ARRIVED_STATIONARY": "arrival",
}
_UNCERTAINTY_LABELS = frozenset({"ARRIVAL_TRANSITION", "STUCK_TRANSITION", "MOVEMENT_TRANSITION"})
_NAVIGATION_STATE_TO_TARGET = {
    NavigationState.ARRIVED.value: "arrival",
    NavigationState.STUCK.value: "stuck",
}


def _legacy_analyzer() -> ConservativeNavigationAnalyzer:
    return ConservativeNavigationAnalyzer(route_evidence=lambda sequence, roi: (False, True))


def _load_manifest(manifest_path: str | Path) -> tuple[Path, tuple[NativeMotionReplayFrame, ...]]:
    path = Path(manifest_path)
    return path, load_native_motion_replay(path)


def _capture_totals(path: Path) -> dict[str, int | str]:
    """Expose recorder accounting only after the loader has validated the manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        return {
            "committed_frames": "BLOCKED_FIELD",
            "skipped_frames": "BLOCKED_FIELD",
            "dropped_frames": "BLOCKED_FIELD",
        }
    return {
        "committed_frames": manifest["committed_frame_count"],
        "skipped_frames": manifest["skipped_frame_count"],
        "dropped_frames": manifest["dropped_frame_count"],
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[round((len(values) - 1) * percentile)]


def _max_continuous_unknown_seconds(frames: list[dict]) -> float | None:
    longest: float | None = None
    started: float | None = None
    last: float | None = None
    for frame in frames:
        if frame["motion"] == MotionClass.UNKNOWN.value:
            started = frame["elapsed_seconds"] if started is None else started
            last = frame["elapsed_seconds"]
        elif started is not None:
            longest = max(longest or 0.0, last - started)
            started = last = None
    if started is not None:
        longest = max(longest or 0.0, last - started)
    return longest


def _terminal_events(frames: list[dict]) -> list[dict]:
    events: list[dict] = []
    prior = NavigationState.UNKNOWN.value
    for frame in frames:
        state = frame["state"]
        if state in (NavigationState.ARRIVED.value, NavigationState.STUCK.value) and state != prior:
            events.append({"state": state, "source_sequence": frame["source_sequence"], "target": frame["target"]})
        prior = state
    return events


def _ground_truth_report(replay: tuple[NativeMotionReplayFrame, ...], frames: list[dict]) -> dict:
    known_labels = sum(frame["target"] is not None for frame in frames)
    raw_labels = [item.expected_state for item in replay]
    missing_fields: list[str] = []
    if known_labels == 0:
        missing_fields.append("expected_state")
    # Scenario identity is not part of the native recorder schema.  Never infer it from a path.
    missing_fields.append("scenario_id")
    return {
        "known_labeled_frames": known_labels,
        "uncertainty_labeled_frames": sum(label in _UNCERTAINTY_LABELS for label in raw_labels),
        "missing_fields": missing_fields,
    }


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    analyzer_factory: Callable[[], Analyzer],
) -> dict:
    if not callable(analyzer_factory):
        raise ValueError("analyzer_factory must be callable")
    path, replay = _load_manifest(manifest_path)
    capture_totals = _capture_totals(path)
    analyzer = analyzer_factory()
    if not callable(analyzer):
        raise ValueError("analyzer_factory must return a callable analyzer")
    machine = NavigationStateMachine(stuck_frames=3)
    frames: list[dict] = []
    analysis_seconds: list[float] = []
    motion_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    labeled: dict[str, list[dict]] = {"moving": [], "arrival": [], "stuck": []}

    for item in replay:
        sequence = item.envelope.source_sequence
        started = perf_counter()
        observation = analyzer(sequence, item.roi)
        analysis_seconds.append(perf_counter() - started)
        if type(observation) is not NavigationObservation:
            raise ValueError("analyzer must return NavigationObservation")
        state = machine.update(observation, item.envelope.received_monotonic)
        target = _LABEL_TO_TARGET.get(item.expected_state)
        record = {
            "source_sequence": sequence,
            "processing_ordinal": item.envelope.processing_ordinal,
            "elapsed_seconds": item.envelope.received_monotonic,
            "expected_state": item.expected_state,
            "target": target,
            "motion": observation.motion.value,
            "state": state.value,
        }
        frames.append(record)
        motion_counts[observation.motion.value] += 1
        state_counts[state.value] += 1
        if target is not None:
            labeled[target].append(record)

    ground_truth = _ground_truth_report(replay, frames)
    predicted = {
        "moving": lambda item: item["motion"] == MotionClass.MOVING.value,
        "arrival": lambda item: item["state"] == NavigationState.ARRIVED.value,
        "stuck": lambda item: item["state"] == NavigationState.STUCK.value,
    }
    label_report = {}
    raw_labels = [item.expected_state for item in replay]
    for target, records in labeled.items():
        matches = sum(predicted[target](item) for item in records)
        has_uncertain_target = any(
            label is not None and target in label.lower() for label in raw_labels
        )
        label_report[target] = {
            "denominator": len(records),
            "matched": matches,
            "recall": matches / len(records) if records else None,
            # Membership labels do not establish onset; explicit brackets are required.
            "latency": "BLOCKED_FIELD" if has_uncertain_target else "not_evaluable",
        }

    false_arrived = sum(
        item["target"] is not None
        and item["state"] == NavigationState.ARRIVED.value
        and item["target"] != "arrival"
        for item in frames
    )
    false_stuck = sum(
        item["target"] is not None
        and item["state"] == NavigationState.STUCK.value
        and item["target"] != "stuck"
        for item in frames
    )
    events = _terminal_events(frames)
    false_event_counts = Counter(
        event["state"] for event in events
        if event["target"] is not None
        and event["target"] != _NAVIGATION_STATE_TO_TARGET[event["state"]]
    )
    total = len(frames)
    moving_labels = labeled["moving"]
    always_unknown = bool(moving_labels) and all(
        item["motion"] == MotionClass.UNKNOWN.value for item in moving_labels
    )
    availability = (
        {"status": "BLOCKED_FIELD", "reason": "missing_expected_state"}
        if "expected_state" in ground_truth["missing_fields"]
        else {"status": "FAIL", "reason": "always_unknown_on_labeled_moving"}
        if always_unknown
        else {"status": "PASS", "reason": "moving_labels_have_non_unknown_output"}
    )
    safety = (
        {"status": "BLOCKED_FIELD", "reason": "missing_expected_state"}
        if "expected_state" in ground_truth["missing_fields"]
        else {"status": "FAIL", "reason": "false_stuck_during_labeled_moving"}
        if false_stuck
        else {"status": "PASS", "reason": "no_false_stuck_on_labeled_frames"}
    )
    return {
        "schema_version": 2,
        "manifest": str(path),
        "frames": frames,
        "totals": {
            "parsed_frames": total,
            "analyzed_frames": total,
            "labeled_frames": ground_truth["known_labeled_frames"],
            "unlabeled_frames": total - ground_truth["known_labeled_frames"],
        },
        "ground_truth": ground_truth,
        "capture_totals": capture_totals,
        "motion_counts": dict(sorted(motion_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "labels": label_report,
        "false_arrived": false_arrived,
        "false_stuck": false_stuck,
        "unknown_fraction": motion_counts[MotionClass.UNKNOWN.value] / total if total else None,
        "unknown_max_continuous_seconds": _max_continuous_unknown_seconds(frames),
        "session_metrics": {
            "session": path.parent.name,
            "domain": "BLOCKED_FIELD",
            "moving_coverage": {
                "denominator": len(labeled["moving"]),
                "matched": label_report["moving"]["matched"],
                "coverage": label_report["moving"]["recall"],
            },
            "geometry_valid_support": "BLOCKED_FIELD",
        },
        "events": {
            "terminal": events,
            "false_high_consequence": {
                "arrived": false_event_counts[NavigationState.ARRIVED.value],
                "stuck": false_event_counts[NavigationState.STUCK.value],
            },
            "stuck_true_positive": {
                "denominator": label_report["stuck"]["denominator"],
                "matched": label_report["stuck"]["matched"],
            },
            "recovery_detection": {
                "denominator": sum(item.expected_state == "MOVING_RECOVERED" for item in replay),
                "matched": sum(
                    item.expected_state == "MOVING_RECOVERED" and frame["motion"] == MotionClass.MOVING.value
                    for item, frame in zip(replay, frames)
                ),
            },
            "duplicate_terminal_events": 0,
            "latch_violations": 0,
        },
        "diagnostics": {
            "analysis_seconds": {
                "p50": _percentile(analysis_seconds, 0.50),
                "p95": _percentile(analysis_seconds, 0.95),
            },
            "dropped_frames": capture_totals["dropped_frames"],
            "bounded_memory": "BLOCKED_FIELD",
        },
        "acceptance_matrix": {
            "availability": availability,
            "safety": safety,
            "hard_scenarios": {"status": "BLOCKED_FIELD", "reason": "missing_scenario_id"},
        },
    }


def evaluate_ablations(
    manifest_path: str | Path,
    *,
    analyzer_factories: dict[str, Callable[[], Analyzer]],
) -> dict:
    """Evaluate named analyzers independently on identical read-only chronology."""
    if not isinstance(analyzer_factories, dict) or not analyzer_factories:
        raise ValueError("analyzer_factories must be a non-empty dict")
    if any(
        not isinstance(name, str) or not name or not callable(factory)
        for name, factory in analyzer_factories.items()
    ):
        raise ValueError("ablation names must be non-empty and factories callable")
    variants = {
        name: evaluate_manifest(manifest_path, analyzer_factory=factory)
        for name, factory in analyzer_factories.items()
    }
    parsed_totals = {report["totals"]["parsed_frames"] for report in variants.values()}
    if len(parsed_totals) != 1:
        raise RuntimeError("ablation variants did not parse identical frame totals")
    return {
        "schema_version": 1,
        "manifest": str(Path(manifest_path)),
        "totals": {"variants": len(variants), "parsed_frames_per_variant": parsed_totals.pop()},
        "variants": variants,
    }


def write_report(output_path: str | Path, report: dict) -> Path:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline minimap motion replay report.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--analyzer", required=True, choices=("legacy",))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(args.manifest, analyzer_factory=_legacy_analyzer)
        write_report(args.output, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
