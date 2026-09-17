"""Render fixed-star references and alpha overlays; this does not generate artwork.

Requires Pillow. Run from any working directory. Star geometry and short great-circle
edges come from the saved draw-9 round, not from replaying the current generator.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SIZE, SS = 1024, 3
TARGETS = ("Z01", "Z11", "Z13", "Z15")
BG = (16, 25, 35, 255)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def projector(center):
    lon, lat = map(math.radians, (center["longitude"], center["latitude"]))
    axis = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    east = (-math.sin(lon), math.cos(lon), 0)
    north = (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat))

    def project(v):
        scale = 2 / (1 + dot(v, axis)) * 180 / math.pi
        return -dot(v, east) * scale, dot(v, north) * scale

    return project


def draw_region(region):
    stars = {s["label"]: s for s in region["stars"]}
    variant = next(v for v in region["variants"] if v["id"] == "extended")
    xs, ys = [s["x"] for s in stars.values()], [s["y"] for s in stars.values()]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.3 + 3
    project = projector(region["projectionCenter"])

    def pixel(xy):
        return (0.5 + (xy[0] - cx) / span) * SIZE, (0.5 - (xy[1] - cy) / span) * SIZE

    def high(xy):
        return tuple(v * SS for v in pixel(xy))

    overlay = Image.new("RGBA", (SIZE * SS, SIZE * SS))
    draw = ImageDraw.Draw(overlay)
    for a, b in variant["edges"]:
        u, v = stars[a]["direction"], stars[b]["direction"]
        angle = math.acos(max(-1, min(1, dot(u, v))))
        assert 0 < angle < math.pi
        steps = max(8, math.ceil(math.degrees(angle) * 5))
        curve = []
        for i in range(steps + 1):
            t = i / steps
            p = [(math.sin((1 - t) * angle) * x + math.sin(t * angle) * y) / math.sin(angle)
                 for x, y in zip(u, v)]
            curve.append(high(project(p)))
        draw.line(curve, fill=(110, 214, 206, 195), width=2 * SS, joint="curve")

    anchors = []
    for s in stars.values():
        q = project(s["direction"])
        assert max(abs(q[0] - s["x"]), abs(q[1] - s["y"])) < 1e-8
        px, py = pixel(q)
        radius = max(2.6, 3 + (5 - s["app_mag"]) * 0.85)
        x, y, r = px * SS, py * SS, radius * SS
        draw.ellipse((x - r, y - r, x + r, y + r), fill=s["color_hex"])
        anchors.append({"label": s["label"], "id": s["id"], "x": px, "y": py,
                        "app_mag": s["app_mag"], "color_hex": s["color_hex"]})

    small_overlay = overlay.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    small_overlay.save(ROOT / "generation" / f'{region["id"]}-overlay.png')
    reference = Image.alpha_composite(Image.new("RGBA", overlay.size, BG), overlay)
    draw = ImageDraw.Draw(reference)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20 * SS)
    except OSError:
        font = ImageFont.load_default(size=20 * SS)
    for a in anchors:
        draw.text(((a["x"] + 11) * SS, (a["y"] - 26) * SS), a["label"],
                  font=font, fill="#e3edf5")
    reference = reference.resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert("RGB")
    reference.save(ROOT / "generation" / f'{region["id"]}-reference.png')
    return {"id": region["id"], "stars": anchors, "edges": variant["edges"],
            "scalePixelsPerDegree": SIZE / span, "center": [cx, cy]}


if __name__ == "__main__":
    data = json.loads((ROOT / "shape-manifest.json").read_text())
    (ROOT / "generation").mkdir(exist_ok=True)
    result = {
        "sourceSha256": data["sourceSha256"],
        "canvas": {"width": SIZE, "height": SIZE, "origin": "top-left", "yAxis": "down"},
        "orientation": data["orientation"],
        "note": "Fixed star and line overlays, not AI illustrations. Each region is fitted with uniform scale.",
        "regions": [draw_region(r) for r in data["regions"] if r["id"] in TARGETS],
    }
    (ROOT / "generation" / "anchors.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Rendered four labeled references and four transparent overlays; verified all 35 projected star positions.")
