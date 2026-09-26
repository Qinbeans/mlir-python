"""Write a flat package index of the wheels attached to GitHub releases.

The index is one HTML page of links to wheel files: what ``pip --find-links``
and uv's ``format = "flat"`` indexes read. Wheels stay on the releases; each
link carries the asset's SHA-256 so installers verify what they download.
Several repositories' releases can go into one index.

Usage::

    for repo in OWNER/REPO OWNER/OTHER; do
        gh api --paginate --slurp "repos/$repo/releases"
    done | python tools/flat_index.py > index.html
"""

import html
import json
import sys
from typing import Any


def wheel_links(releases: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(filename, url) for every wheel on a published release, newest first."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for release in releases:
        if release["draft"]:
            continue
        for asset in release["assets"]:
            name: str = asset["name"]
            if not name.endswith(".whl") or name in seen:
                continue
            seen.add(name)
            url: str = asset["browser_download_url"]
            digest: str | None = asset.get("digest")
            if digest and digest.startswith("sha256:"):
                url += "#sha256=" + digest.removeprefix("sha256:")
            links.append((name, url))
    return links


def render(links: list[tuple[str, str]]) -> str:
    """The index page, with the wheels grouped by package."""
    packages: dict[str, list[tuple[str, str]]] = {}
    for name, url in links:
        packages.setdefault(name.split("-")[0].replace("_", "-"), []).append(
            (name, url)
        )
    sections = "\n".join(
        f"<h2>{html.escape(package)}</h2>\n"
        + "\n".join(
            f'<a href="{html.escape(url)}">{html.escape(name)}</a><br>'
            for name, url in wheels
        )
        for package, wheels in sorted(packages.items())
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Package index</title></head>
<body>
<h1>Package index</h1>
{sections}
</body>
</html>
"""


def read_releases(text: str) -> list[dict[str, Any]]:
    """Every release in ``text``: one or more JSON documents, each a list of
    releases or, from ``gh api --paginate --slurp``, a list of pages."""
    decoder = json.JSONDecoder()
    releases: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        data, position = decoder.raw_decode(text, position)
        for item in data:
            releases.extend(item if isinstance(item, list) else [item])
    return releases


def main() -> None:
    sys.stdout.write(render(wheel_links(read_releases(sys.stdin.read()))))


if __name__ == "__main__":
    main()
