"""Render numbered star references and package the constellation prompts.

Requires Pillow. --refresh updates the current pack after verifying that every
unlabelled overlay is unchanged. Historical reference sets are never overwritten.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from build_references import BG, ROOT, SIZE, draw_region

PACK = ROOT / "constellations"
REPORTS = ROOT.parents[2] / "reports" / "zodiac_imagery" / ROOT.name
IDS = [f"Z{i:02d}" for i in range(1, 16)]


def get_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def number_reference(alpha, anchors):
    """Put local numeric labels in empty space without covering any star or edge."""
    image = Image.alpha_composite(Image.new("RGBA", alpha.size, BG), alpha).convert("RGB")
    draw = ImageDraw.Draw(image)
    occupied = alpha.getchannel("A").filter(ImageFilter.MaxFilter(7))
    mask_draw = ImageDraw.Draw(occupied)
    font = get_font(22)
    numbered = [dict(a, number=f"{i:02d}")
                for i, a in enumerate(sorted(anchors, key=lambda a: a["label"]), 1)]

    # Give tightly spaced pairs first choice of the available label positions.
    def closest_distance(a):
        return min(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2
                    for b in numbered if b is not a), default=float("inf"))

    for a in sorted(numbered, key=closest_distance):
        text = a["number"]
        bbox = draw.textbbox((0, 0), text, font=font, anchor="lt")
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = a["x"], a["y"]
        placed = False
        for gap in (13, 20, 28, 40, 56, 76):
            positions = [(x + gap, y - h / 2), (x - w - gap, y - h / 2),
                         (x - w / 2, y - h - gap), (x - w / 2, y + gap),
                         (x + gap, y - h - gap), (x - w - gap, y - h - gap),
                         (x + gap, y + gap), (x - w - gap, y + gap)]
            for left, top in positions:
                left, top = round(left), round(top)
                box = [left - 3, top - 3, left + w + 3, top + h + 3]
                if min(box[:2]) < 8 or max(box[2:]) > SIZE - 8:
                    continue
                if occupied.crop(box).getbbox() is not None:
                    continue
                center = (left + w / 2, top + h / 2)
                own_distance = (center[0] - x) ** 2 + (center[1] - y) ** 2
                if any((center[0] - b["x"]) ** 2 + (center[1] - b["y"]) ** 2 <= own_distance
                       for b in numbered if b is not a):
                    continue
                draw.text((left - bbox[0], top - bbox[1]), text,
                          font=font, anchor="lt", fill="#e3edf5")
                mask_draw.rectangle(box, fill=255)
                a["labelBox"] = box
                placed = True
                break
            if placed:
                break
        if not placed:
            raise ValueError(f"No clear label position for {a['label']}")
    return image, numbered


def main(refresh=False):
    data = json.loads((ROOT / "shape-manifest.json").read_text())
    regions = {r["id"]: r for r in data["regions"]}
    assert len(data["regions"]) == len(IDS) and set(regions) == set(IDS)
    REPORTS.mkdir(parents=True, exist_ok=True)
    archive = REPORTS / "Terrax-星座包.zip"
    if archive.exists() and not refresh:
        raise FileExistsError(f"Archive already exists; use --refresh to update: {archive}")
    for name in IDS:
        for path in (PACK / f"{name}.png", PACK / "transparent" / f"{name}.png"):
            if path.exists() and not refresh:
                raise FileExistsError(path)
            if refresh and not path.exists():
                raise FileNotFoundError(path)

    audit = {"sourceSha256": data["sourceSha256"],
             "regionSources": data.get("regionSources", {}),
             "sourceNote": data.get("regionSourceNote", ""),
             "numbering": "Local 01..N in original letter-label order; reference PNGs only.",
             "regions": []}
    contact = Image.new("RGB", (2400, 1584), BG[:3])
    draw = ImageDraw.Draw(contact)
    font = get_font(28)

    with TemporaryDirectory(prefix=".terrax-constellation-pack-", dir=REPORTS) as tmp:
        staged = Path(tmp) / "constellations"
        (staged / "transparent").mkdir(parents=True)
        for i, name in enumerate(IDS):
            region = regions[name]
            variant = next(v for v in region["variants"] if v["id"] == "extended")
            members = {s["label"] for s in region["stars"]}
            assert members == set(variant["members"])
            assert all(a in members and b in members for a, b in variant["edges"])
            rendered = draw_region(region, output_dir=tmp)
            assert rendered["edges"] == variant["edges"]
            assert [s["id"] for s in rendered["stars"]] == [s["id"] for s in region["stars"]]
            alpha_path = Path(tmp) / f"{name}-overlay.png"
            with Image.open(alpha_path) as alpha:
                assert alpha.size == (SIZE, SIZE) and alpha.mode == "RGBA"
                assert alpha.getchannel("A").getextrema() == (0, 255)
                image, anchors = number_reference(alpha, rendered["stars"])
                image.save(staged / f"{name}.png")
            if refresh and alpha_path.read_bytes() != (PACK / "transparent" / f"{name}.png").read_bytes():
                raise ValueError(f"{name}: saved overlay differs; refusing to change its geometry")
            shutil.copyfile(alpha_path, staged / "transparent" / f"{name}.png")
            x, y = i % 5 * 480, i // 5 * 528
            contact.paste(image.resize((480, 480), Image.Resampling.LANCZOS), (x, y + 48))
            draw.text((x + 18, y + 10), name, font=font, fill="#e3edf5")
            audit["regions"].append({
                "id": name, "stars": len(members), "edges": len(variant["edges"]),
                "isolated": variant["graph"]["isolated"], "anchors": anchors,
                "referenceSha256": hashlib.sha256((staged / f"{name}.png").read_bytes()).hexdigest(),
                "transparentSha256": hashlib.sha256(alpha_path.read_bytes()).hexdigest(),
            })

        for name in ("README.md", "prompts.md"):
            shutil.copyfile(PACK / name, staged / name)
        files = sorted(p for p in staged.rglob("*") if p.is_file())
        staged_archive = Path(tmp) / archive.name
        with ZipFile(staged_archive, "x", compression=ZIP_DEFLATED) as z:
            for path in files:
                z.write(path, Path("constellations") / path.relative_to(staged))
        with ZipFile(staged_archive) as z:
            assert z.testzip() is None and len(z.namelist()) == 32
            for path in files:
                assert z.read(str(Path("constellations") / path.relative_to(staged))) == path.read_bytes()

        # Publish only after all fifteen original overlays and the ZIP have passed.
        for path in files:
            rel = path.relative_to(staged)
            for destination in (PACK / rel, REPORTS / "constellations" / rel):
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or destination.read_bytes() != path.read_bytes():
                    shutil.copyfile(path, destination)
        os.replace(staged_archive, archive)
        contact.save(REPORTS / "星座包总览.png")
        audit["totals"] = {"constellations": len(IDS),
                           "members": sum(r["stars"] for r in audit["regions"]),
                           "edges": sum(r["edges"] for r in audit["regions"])}
        (REPORTS / "constellation-pack-validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"Extracted {len(IDS)} constellations, {sum(r['stars'] for r in audit['regions'])} stars, "
          f"{sum(r['edges'] for r in audit['regions'])} saved edges.")
    print(f"Verified 15 numbered references + 15 unlabelled alpha PNGs and 32 archive members: {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Update the current pack while preserving all alpha PNGs")
    main(refresh=parser.parse_args().refresh)
