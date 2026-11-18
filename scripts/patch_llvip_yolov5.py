from pathlib import Path


YOLO_FILE = Path("vendor/LLVIP/yolov5/models/yolo.py")


PATCHES = (
    (
        "    def forward(self, x):\n"
        "        z = []  # inference output\n",
        "    def forward(self, x):\n"
        "        z = []  # inference output\n"
        "        if not isinstance(self.grid, list):\n"
        "            if 'grid' in self._buffers:\n"
        "                del self._buffers['grid']\n"
        "            self.grid = [torch.zeros(1, device=x[0].device)] * self.nl\n"
        "        if not isinstance(self.anchor_grid, list):\n"
        "            if 'anchor_grid' in self._buffers:\n"
        "                del self._buffers['anchor_grid']\n"
        "            self.anchor_grid = [torch.zeros(1, device=x[0].device)] * self.nl\n",
    ),
    (
        "    def forward(self, x, augment=False, profile=False, visualize=False):\n",
        "    def forward(self, x, augment=False, profile=False, visualize=False, embed=None):\n",
    ),
    (
        "    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers\n"
        "        LOGGER.info('Fusing layers... ')\n",
        "    def fuse(self, verbose=True):  # fuse model Conv2d() + BatchNorm2d() layers\n"
        "        if verbose:\n"
        "            LOGGER.info('Fusing layers... ')\n",
    ),
    (
        "        self.info()\n"
        "        return self\n"
        "\n"
        "    def autoshape",
        "        if verbose:\n"
        "            self.info()\n"
        "        return self\n"
        "\n"
        "    def autoshape",
    ),
)


def main():
    if not YOLO_FILE.is_file():
        raise SystemExit(f"LLVIP YOLOv5 file not found: {YOLO_FILE}")

    text = YOLO_FILE.read_text()
    changed = False
    for old, new in PATCHES:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(f"Expected text not found in {YOLO_FILE}")
        text = text.replace(old, new, 1)
        changed = True

    if changed:
        YOLO_FILE.write_text(text)
        print(f"Patched {YOLO_FILE}")
    else:
        print(f"Already patched: {YOLO_FILE}")


if __name__ == "__main__":
    main()
