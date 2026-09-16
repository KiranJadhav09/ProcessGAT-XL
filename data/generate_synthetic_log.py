"""
Generates a synthetic loan application event log in XES format,
modelled on the BPIC 2012 structure.

Activities mirror the real BPIC 2012 process:
  A_ prefix = application activities
  O_ prefix = offer activities
  W_ prefix = work-item activities

Approved cases (label 1) follow a structured path.
Rejected/cancelled cases (label 0) terminate early or take decline paths.

Run once to produce: data/synthetic_loan.xes
"""


import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# ── Process definition ────────────────────────────────────────────────────────

APPROVED_PATHS = [
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
     "W_Completeren aanvraag", "A_ACCEPTED", "O_SELECTED",
     "A_FINALIZED", "O_CREATED", "O_SENT", "W_Nabellen offertes",
     "O_SENT_BACK", "W_Valideren aanvraag", "A_APPROVED",
     "O_ACCEPTED", "A_ACTIVATED"],
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
     "W_Completeren aanvraag", "A_ACCEPTED", "O_SELECTED",
     "A_FINALIZED", "O_CREATED", "O_SENT", "O_SENT_BACK",
     "W_Valideren aanvraag", "A_APPROVED", "O_ACCEPTED", "A_ACTIVATED"],
]

DECLINED_PATHS = [
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_DECLINED"],
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
     "W_Completeren aanvraag", "A_ACCEPTED", "O_SELECTED",
     "A_FINALIZED", "O_CREATED", "O_SENT", "O_CANCELLED"],
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
     "W_Completeren aanvraag", "A_ACCEPTED", "A_CANCELLED"],
    ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
     "W_Completeren aanvraag", "W_Nabellen incomplete dossiers",
     "A_DECLINED"],
]


def make_trace(case_id, path, start_time):
    events = []
    t = start_time
    for activity in path:
        t += timedelta(minutes=random.randint(5, 480))
        events.append({
            "concept:name": activity,
            "time:timestamp": t.isoformat(),
            "org:resource": f"resource_{random.randint(1, 20):02d}",
        })
    return {"case_id": str(case_id), "events": events}


def generate_log(n_cases=2000, approved_ratio=0.35):
    traces = []
    start = datetime(2012, 1, 1, 8, 0, 0)
    for i in range(n_cases):
        case_start = start + timedelta(hours=random.randint(0, 8760))
        if random.random() < approved_ratio:
            path = random.choice(APPROVED_PATHS)
            # add some noise events with 30% probability
            if random.random() < 0.3:
                noise = random.choice(["W_Nabellen offertes",
                                       "W_Completeren aanvraag"])
                insert_at = random.randint(1, len(path) - 1)
                path = path[:insert_at] + [noise] + path[insert_at:]
        else:
            path = random.choice(DECLINED_PATHS)
        traces.append(make_trace(i + 1, path, case_start))
    return traces


# ── XES writer ────────────────────────────────────────────────────────────────

def write_xes(traces, path):
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" ?>')
    lines.append('<log xes.version="1.0" xes.features="nested-attributes" '
                 'openxes.version="1.0RC7">')
    lines.append('  <extension name="Concept" prefix="concept" '
                 'uri="http://www.xes-standard.org/concept.xesext"/>')
    lines.append('  <extension name="Time" prefix="time" '
                 'uri="http://www.xes-standard.org/time.xesext"/>')
    lines.append('  <extension name="Organizational" prefix="org" '
                 'uri="http://www.xes-standard.org/org.xesext"/>')

    for trace in traces:
        lines.append('  <trace>')
        lines.append(f'    <string key="concept:name" value="{trace["case_id"]}"/>')
        for event in trace["events"]:
            lines.append('    <event>')
            lines.append(f'      <string key="concept:name" '
                         f'value="{event["concept:name"]}"/>')
            lines.append(f'      <date key="time:timestamp" '
                         f'value="{event["time:timestamp"]}"/>')
            lines.append(f'      <string key="org:resource" '
                         f'value="{event["org:resource"]}"/>')
            lines.append('    </event>')
        lines.append('  </trace>')

    lines.append('</log>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Written: {path}")
    print(f"Cases: {len(traces)}")
    print(f"Events: {sum(len(t['events']) for t in traces)}")
    approved = sum(
        1 for t in traces
        if any(e["concept:name"] == "A_APPROVED" for e in t["events"])
    )
    print(f"Approved: {approved} ({100*approved/len(traces):.1f}%)")


if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_loan.xes")
    traces = generate_log(n_cases=2000, approved_ratio=0.35)
    write_xes(traces, out_path)