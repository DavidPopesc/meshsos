"""Build docs/ screenshots and the README GIF from viewer capture frames.

1. python -m meshsos view --capture-dir docs/frames --no-browser
2. open, in a WebGL browser:
   /viewer/index.html?run=/run/timeline.json&capture=1&frames=90&from=116&to=206&fps=10&layout=quad&fly=1&w=800&h=450&prefix=gif
   /viewer/index.html?run=/run/timeline.json&capture=1&at=178&layout=quad&w=1600&h=900&prefix=viewer_quad
   /viewer/index.html?run=/run/timeline.json&capture=1&at=186&variant=compressed_agent&cam=gateway&w=1600&h=900&prefix=viewer_closeup&speed=6
3. python scripts/make_gif.py        (needs Pillow: pip install pillow)
"""

import glob
import sys
from pathlib import Path

from PIL import Image

FRAMES = Path("docs/frames")


def main() -> int:
    files = sorted(glob.glob(str(FRAMES / "gif_*.png")))[::2]
    if not files:
        print("no docs/frames/gif_*.png - capture frames first (see the docstring)")
        return 1
    frames = [Image.open(f).convert("RGB").resize((640, 360), Image.LANCZOS) for f in files]
    palette = frames[len(frames) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save("docs/viewer.gif", save_all=True, append_images=quantized[1:], duration=180, loop=0, optimize=True)
    for name in ("viewer_quad", "viewer_closeup"):
        src = FRAMES / f"{name}.png"
        if src.exists():
            Image.open(src).convert("RGB").save(f"docs/{name}.jpg", quality=88)
    print(f"wrote docs/viewer.gif ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
