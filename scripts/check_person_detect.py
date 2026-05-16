from pathlib import Path
import re

import numpy as np
from PIL import Image
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "person_detect.tflite"
SAMPLES = {
    "person": ROOT / "samples" / "person.bmp",
    "no_person": ROOT / "samples" / "no_person.bmp",
}
FIXTURE = ROOT / "ember-infer-ref" / "tests" / "fixtures" / "person_detect.rs"


def run_variant(name: str, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = arr.astype(input_details["dtype"]).reshape(input_details["shape"])
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"]).reshape(-1)

    scale, zero_point = output_details["quantization"]
    if scale:
        scores = (out.astype(np.int32) - zero_point) * scale
    else:
        scores = out.astype(np.float32)
    return out, scores


def describe_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    print(f"[{path.name}] size={img.size} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
    return arr


def extract_fixture(name: str) -> np.ndarray:
    text = FIXTURE.read_text(encoding="utf-8")
    pattern = rf"pub const {name}:.*?= \[matrix!\[(.*?)\]\];"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise RuntimeError(f"fixture {name} not found")
    values = np.array([int(x) for x in re.findall(r"-?\d+", match.group(1))], dtype=np.int16)
    return values.reshape(96, 96)


def main() -> None:
    base_images = {name: describe_image(path) for name, path in SAMPLES.items()}

    variants = {
        "u8_raw": lambda a: a,
        "i8_shift_128": lambda a: a.astype(np.int16) - 128,
        "i8_invert_shift_128": lambda a: (255 - a).astype(np.int16) - 128,
    }

    for fixture_name, sample_name in [("PERSON", "person"), ("NO_PERSON", "no_person")]:
        fixture = extract_fixture(fixture_name)
        expected = variants["i8_shift_128"](base_images[sample_name])
        print(
            f"fixture_check {fixture_name}: "
            f"equal_to_{sample_name}_bmp={np.array_equal(fixture, expected)} "
            f"diff_count={int(np.count_nonzero(fixture != expected))}"
        )
        for variant_name, fn in variants.items():
            candidate = fn(base_images[sample_name])
            print(
                f"  against {variant_name:20s} "
                f"equal={np.array_equal(fixture, candidate)} "
                f"first_fixture={fixture.flatten()[:12].tolist()} "
                f"first_candidate={candidate.flatten()[:12].tolist()}"
            )

    try:
        probe = tf.lite.Interpreter(model_path=str(MODEL))
        probe.allocate_tensors()
        input_details = probe.get_input_details()[0]
        output_details = probe.get_output_details()[0]
        print(
            "input_details:",
            input_details["shape"],
            input_details["dtype"],
            input_details["quantization"],
        )
        print(
            "output_details:",
            output_details["shape"],
            output_details["dtype"],
            output_details["quantization"],
        )
    except Exception as exc:
        print(f"model_load_error: {exc}")
        return

    for sample_name, arr in base_images.items():
        print(f"\n== sample: {sample_name} ==")
        for variant_name, fn in variants.items():
            raw, scores = run_variant(variant_name, fn(arr))
            top = int(np.argmax(scores))
            print(
                f"{variant_name:20s} raw={raw.tolist()} scores={scores.tolist()} top={top}"
            )


if __name__ == "__main__":
    main()
