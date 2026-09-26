import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from phash_match import ImageHash, average_hash, compare, difference_hash, find_duplicates, phash
from phash_match.cli import main

THRESHOLD = 10


def scene(seed: int, size=(640, 480)) -> Image.Image:
    """A photo-like synthetic image: gradient background plus random shapes."""
    rng = np.random.default_rng(seed)
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * rng.uniform(0.1, 0.5) + y * rng.uniform(0.1, 0.5) + rng.uniform(0, 80)) % 256 for _ in range(3)], -1)
    img = Image.fromarray(base.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        box = [x0, y0, x0 + int(rng.integers(40, 250)), y0 + int(rng.integers(40, 200))]
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)(box, fill=color)
    return img


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


EDITS = {
    "resized_to_25pct": lambda im: im.resize((im.width // 4, im.height // 4)),
    "jpeg_quality_20": lambda im: jpeg(im, 20),
    "brighter_30pct": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    "more_contrast": lambda im: ImageEnhance.Contrast(im).enhance(1.4),
    "slight_blur": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "grayscale": lambda im: im.convert("L"),
    "resized_and_recompressed": lambda im: jpeg(im.resize((320, 240)), 40),
}


@pytest.mark.parametrize("edit", EDITS)
@pytest.mark.parametrize("seed", range(5))
def test_edited_copies_still_match(seed, edit):
    original = scene(seed)
    assert phash(original) - phash(EDITS[edit](original)) <= THRESHOLD


def test_different_images_do_not_match():
    hashes = [phash(scene(s)) for s in range(40)]
    distances = [a - b for i, a in enumerate(hashes) for b in hashes[i + 1:]]
    assert min(distances) > THRESHOLD


def test_identical_image_distance_zero():
    img = scene(1)
    for fn in (phash, average_hash, difference_hash):
        assert fn(img) - fn(img.copy()) == 0


def test_matches_imagehash_reference():
    """Our pHash / aHash / dHash equal the widely used `imagehash` library's on the same inputs."""
    imagehash = pytest.importorskip("imagehash")
    for s in range(20):
        img = scene(s)
        assert str(phash(img)) == str(imagehash.phash(img))
        assert str(average_hash(img)) == str(imagehash.average_hash(img))
        assert str(difference_hash(img)) == str(imagehash.dhash(img))


def test_hex_roundtrip():
    h = phash(scene(3))
    assert ImageHash.from_hex(str(h)) == h and len(str(h)) == 16


def test_find_duplicates_in_folder(tmp_path):
    for s in range(6):
        scene(s).save(tmp_path / f"original_{s}.png")
    EDITS["resized_and_recompressed"](scene(2)).save(tmp_path / "copy_of_2.jpg")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"not really a jpeg")
    matches = find_duplicates(tmp_path, THRESHOLD)
    assert [(m.a.name, m.b.name) for m in matches] == [("copy_of_2.jpg", "original_2.png")]


def test_cli(tmp_path, capsys):
    a, b, c = tmp_path / "a.png", tmp_path / "b.jpg", tmp_path / "c.png"
    scene(7).save(a)
    jpeg(scene(7), 30).save(b)
    scene(8).save(c)
    assert main(["compare", str(a), str(b)]) == 0 and "MATCH" in capsys.readouterr().out
    assert main(["compare", str(a), str(c)]) == 1 and "DIFFERENT" in capsys.readouterr().out
    assert main(["scan", str(tmp_path)]) == 0 and "1 near-duplicate pair(s)" in capsys.readouterr().out
    assert main(["hash", str(a)]) == 0
    assert main(["compare", str(a), str(tmp_path / "missing.png")]) == 2
