#!/usr/bin/env python3
"""Build the Formula 110 presentation.

Assembles parts/ + data/deck-data.json + fonts/ into two files:

  index.html     standalone page; double-click it to present offline
  artifact.html  body-only fragment for publishing as a claude.ai Artifact

Standard library only. From the repository root:

  python3 presentation/build.py
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = "The 7.19-Second Lap"
FONTS = (
    ("CMU Serif", "cmunrm", "normal", 400),
    ("CMU Serif", "cmunti", "italic", 400),
    ("CMU Serif", "cmunbx", "normal", 700),
    ("CMU Typewriter", "cmuntt", "normal", 400),
)
SCRIPTS = ("app.js", "viz_track.js", "viz_journey.js", "viz_misc.js")


def read(*parts: str) -> str:
    return HERE.joinpath(*parts).read_text(encoding="utf-8")


def font_faces() -> str:
    rules = []
    for family, name, style, weight in FONTS:
        encoded = base64.b64encode(HERE.joinpath("fonts", name + ".woff").read_bytes()).decode()
        rules.append(
            f"@font-face{{font-family:'{family}';src:url(data:font/woff;base64,{encoded}) format('woff');"
            f"font-style:{style};font-weight:{weight};font-display:swap}}"
        )
    return "\n".join(rules)


def main() -> None:
    style = font_faces() + "\n" + read("parts", "style.css")
    script = (
        "const DATA=" + read("data", "deck-data.json").strip() + ";\n"
        + "\n".join(read("parts", name) for name in SCRIPTS)
        + "\nboot();\n"
    )
    body = read("parts", "body.html")
    fragment = f"<title>{TITLE}</title>\n<style>\n{style}\n</style>\n{body}\n<script>\n{script}</script>\n"
    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{TITLE}</title>\n<style>\n{style}\n</style>\n</head>\n<body>\n{body}\n"
        f"<script>\n{script}</script>\n</body>\n</html>\n"
    )
    HERE.joinpath("artifact.html").write_text(fragment, encoding="utf-8")
    HERE.joinpath("index.html").write_text(page, encoding="utf-8")
    print(f"wrote presentation/index.html ({len(page) // 1024} KB) and presentation/artifact.html")


if __name__ == "__main__":
    main()
