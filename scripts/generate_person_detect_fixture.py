from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "PERSON": ROOT / "samples" / "person.bmp",
    "NO_PERSON": ROOT / "samples" / "no_person.bmp",
}
TARGETS = [
    ROOT / "ember-infer-ref" / "tests" / "fixtures" / "person_detect.rs",
    ROOT / "samples" / "features" / "person_detect.rs",
]
MODE = sys.argv[1] if len(sys.argv) > 1 else "shift-128"
TRANSFORM = sys.argv[2] if len(sys.argv) > 2 else "none"


def load_signed_pixels(path: Path) -> list[list[int]]:
    image = Image.open(path).convert("L")
    if image.size != (96, 96):
        raise ValueError(f"{path} has unexpected size {image.size}, expected (96, 96)")
    rows = []
    for y in range(96):
        row = []
        for x in range(96):
            pixel = int(image.getpixel((x, y)))
            if MODE == "shift-128":
                value = pixel - 128
            elif MODE == "invert-shift-128":
                value = (255 - pixel) - 128
            elif MODE == "wrap-u8":
                value = pixel if pixel < 128 else pixel - 256
            else:
                raise ValueError(f"unsupported mode: {MODE}")
            row.append(value)
        rows.append(row)
    if TRANSFORM == "none":
        return rows
    if TRANSFORM == "flipud":
        return list(reversed(rows))
    if TRANSFORM == "fliplr":
        return [list(reversed(row)) for row in rows]
    if TRANSFORM == "flipud-fliplr":
        return [list(reversed(row)) for row in reversed(rows)]
    raise ValueError(f"unsupported transform: {TRANSFORM}")


def format_const(name: str, rows: list[list[int]]) -> str:
    lines = [
        f"pub const {name}: Buffer4D<i8, 1, 96, 96, 1> = [matrix![",
    ]
    for row in rows:
        body = ", ".join(f"[{value}]" for value in row)
        lines.append(f"    {body};")
    lines.append("]];")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "use nalgebra::matrix;",
        "",
        "use microflow::buffer::Buffer4D;",
        "",
    ]
    for name, path in SAMPLES.items():
        parts.append(format_const(name, load_signed_pixels(path)))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    content = render()
    for target in TARGETS:
        target.write_text(content, encoding="utf-8")
        print(f"wrote {target} with mode={MODE} transform={TRANSFORM}")


if __name__ == "__main__":
    main()
