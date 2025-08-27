import json
import glob
import os
from collections import OrderedDict

# Utility: normalize text for human-readable questions/answers (no underscores)
def norm(text: str) -> str:
    return (text or "").replace("_", " ")

# Compose short textual list of up to three tools for answers
def short_tools_text(tools):
    clean = []
    seen = set()
    for t in tools:
        t_clean = norm(t).strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            clean.append(t_clean)
        if len(clean) >= 3:
            break
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{clean[0]}, {clean[1]}, and {clean[2]}"

# Build three QA pairs per entry with constraints
def build_qa_pairs(entry):
    tools = entry.get("tools", []) or []
    gt = norm(entry.get("groundtruth_taskname", "")).strip() or "the labeled task"

    uses_energy = any(
        any(k in (t or "") for k in [
            "monopolar", "bipolar", "vessel_sealer", "permanent_cautery_hook_spatula", "force_bipolar"
        ]) for t in tools
    )

    qa_pairs = []

    # Q1: Instruments visibility (fallback if no tools)
    if tools:
        q1 = f"Which instruments are visible in this clip while performing {gt}?"
        if len(q1.split()) > 20:
            q1 = "Which instruments are visible in this surgical training video segment?"
        tshort = short_tools_text(tools)
        a1 = [
            f"Includes {tshort}.",
            f"Visible: {tshort}.",
            f"Present: {tshort}.",
            f"Tools include {tshort}.",
            f"Seen here: {tshort}."
        ]
    else:
        q1 = "Are any robotic instruments listed for this segment in the detected objects?"
        a1 = [
            "No, no instruments listed.",
            "No tools are detected.",
            "None listed in objects.",
            "No, instruments not provided.",
            "No detected instruments."
        ]
    qa_pairs.append({"question": q1, "answers": a1})

    # Q2: Energy device usage yes/no
    q2 = "Is an energy device such as monopolar scissors or bipolar forceps being used here?"
    if uses_energy:
        a2 = [
            "Yes, energy device in use.",
            "Yes, monopolar/bipolar used.",
            "Affirmative, energy is used.",
            "Yes, energy instruments present.",
            "Yes, energy applied here."
        ]
    else:
        a2 = [
            "No, no energy device used.",
            "Negative, no energy here.",
            "No energy instruments present.",
            "No, energy not applied.",
            "No, none used here."
        ]
    qa_pairs.append({"question": q2, "answers": a2})

    # Q3: Training objective per ground truth
    q3 = "According to the ground truth label, what training objective is being practiced?"
    gt_short = gt or "the labeled task"
    a3 = [
        f"{gt_short}.",
        f"{gt_short} task.",
        f"Training focus: {gt_short}.",
        f"{gt_short} is practiced.",
        f"Objective: {gt_short}."
    ]
    qa_pairs.append({"question": q3, "answers": a3})

    return qa_pairs


def main():
    # Find all merged metadata files (search recursively to be flexible)
    candidates = sorted(glob.glob("**/merged_objdet_metadata_*.json", recursive=True))
    if not candidates:
        raise SystemExit("No merged_objdet_metadata_*.json files found. Place them in the repository.")

    # Load and merge arrays
    records = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")

    # Build output structure
    out = OrderedDict()
    for item in records:
        case_id = item.get("case_id")
        index = item.get("index")
        if case_id is None or index is None:
            continue
        key = f"{case_id}_id{index}"
        video_path = f"{case_id}_id{index}.mp4"

        entry = {
            "video_path": video_path,
            "detected_objects": item.get("tools", []) or [],
            "qa_pairs": build_qa_pairs(item)
        }
        out[key] = entry

    # Write
    with open("vqa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote vqa_dataset.json with {len(out)} entries from {len(candidates)} metadata files.")


if __name__ == "__main__":
    main()
