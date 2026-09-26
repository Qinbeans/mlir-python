"""Write a flat package index of the wheels attached to GitHub releases.

The index is one HTML page of links to wheel files: what ``pip --find-links``
and uv's ``format = "flat"`` indexes read. Wheels stay on the releases; each
link carries the asset's SHA-256 so installers verify what they download.

Usage::

    gh api --paginate --slurp repos/OWNER/REPO/releases | python tools/flat_index.py > index.html
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
    anchors = "\n".join(
        f'<a href="{html.escape(url)}">{html.escape(name)}</a><br>'
        for name, url in links
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>mlir-python wheels</title></head>
<body>
<h1>mlir-python wheels</h1>
{anchors}
</body>
</html>
"""


def main() -> None:
    data = json.load(sys.stdin)
    # `gh api --paginate --slurp` yields one list per page.
    releases = (
        [release for page in data for release in page]
        if data and isinstance(data[0], list)
        else data
    )
    sys.stdout.write(render(wheel_links(releases)))


if __name__ == "__main__":
    main()
