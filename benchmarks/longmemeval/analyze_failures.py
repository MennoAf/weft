from __future__ import annotations

import codecs
import json
import re
import sys
from collections import defaultdict


def main(result_path: str, dataset_path: str) -> None:
    rows = [json.loads(line) for line in open(result_path) if line.strip()]
    needed = {row["question_id"] for row in rows}
    refs: dict[str, dict] = {}
    decoder = json.JSONDecoder()
    decoder_state = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    started = False
    with open(dataset_path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                buffer += decoder_state.decode(b"", final=True)
                break
            buffer += decoder_state.decode(chunk)
            if not started:
                opening = buffer.find("[")
                if opening < 0:
                    continue
                buffer = buffer[opening + 1 :]
                started = True
            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if not buffer or buffer.startswith("]"):
                    break
                try:
                    item, consumed = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[consumed:]
                if item["question_id"] in needed:
                    refs[item["question_id"]] = item

    def norm(value: object) -> str:
        return re.sub(r"\s+", " ", str(value)).strip()

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    wrong: list[tuple[dict, dict]] = []
    for row in rows:
        reference = refs.get(row["question_id"])
        if reference is None:
            continue
        counts = by_type[reference["question_type"]]
        counts[0] += 1
        if row.get("autoeval_label", {}).get("label"):
            counts[1] += 1
        else:
            wrong.append((row, reference))

    print(f"refs={len(refs)} rows={len(rows)} wrong={len(wrong)}")
    print("BY TYPE")
    for question_type, (total, correct) in sorted(by_type.items()):
        print(f"{question_type}\t{correct}\t{total}\t{correct / total:.4f}")
    print(f"WRONG={len(wrong)}")
    for row, reference in wrong:
        print(f"\n--- {row['question_id']} {reference['question_type']}")
        print(f"Q: {norm(reference['question'])}")
        print(f"GOLD: {norm(reference['answer'])[:500]}")
        print(f"HYP: {norm(row['hypothesis'])[:900]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
