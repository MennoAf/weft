from __future__ import annotations

import codecs
import json
import sys


def main(ids_text: str, source_path: str, output_path: str) -> None:
    wanted = set(ids_text.split(","))
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    started = False
    selected: list[dict] = []
    with open(source_path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                buffer += utf8.decode(b"", final=True)
                break
            buffer += utf8.decode(chunk)
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
                if item["question_id"] in wanted:
                    selected.append(item)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(selected, output)
    print(f"selected={len(selected)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
