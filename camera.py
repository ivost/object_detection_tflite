#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional
import matplotlib.cm as cm
import cv2
import numpy as np
from PIL import Image

FRAME_PER_SECOND = 30


class Camera(object):
    def __init__(
            self: Camera,
            width: int,
            height: int,
            threshold: float,
            fontsize: int,
            fastforward: int = 1
    ) -> None:
        self._dims = (width, height)
        self._default_color = (0xff, 0xff, 0xff)
        self._threshold = threshold
        self._fontsize = fontsize
        self._fastforward = fastforward
        return

    def clear(self: Camera) -> None:
        # print("clear")
        # self._draw.rectangle(
        #     (0, 0) + self._dims,
        #     fill=(0, 0, 0, 0x00)
        # )
        return

    def draw_objects(self: Camera, objects: List) -> None:
        for obj in objects:
            self.draw_object(obj)
        return

    def draw_time(self: Camera, elapsed_ms: float) -> None:
        text = 'Elapsed Time: %.1f[ms]' % elapsed_ms
        if self._fastforward > 1:
            text += ' (speed x%d)' % self._fastforward
        self.draw_text(
            text, location=(10, 10), color=None
        )
        return

    def draw_count(self: Camera, count: int) -> None:
        self.draw_text(
            'Detected Objects: %d' % count,
            location=(5, 5 + self._fontsize), color=None
        )
        return

    def draw_object(self: Camera, object: Dict) -> None:
        prob = object['prob']
        x = (prob - self._threshold) / (1.0 - self._threshold)
        # print(f"prob {prob}, x {x}, cm {cm.jet(x)}")
        color = tuple(np.array(np.array(cm.jet(x)) * 255, dtype=np.uint8).tolist())
        color = color[:-1]

        # print(f"color {color}")
        # print(f"{object}")
        self.draw_box(rect=object['bbox'], color=color)
        name = object.get('name')
        xoff = object['bbox'][0] + 5
        yoff = object['bbox'][1] + 5
        self.draw_text(f"{prob:.2f} {name}", location=(xoff, yoff), color=color)
        return

    def draw_box(
            self: Camera,
            rect: Tuple[int, int, int, int],
            color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        outline = color or self._default_color
        line_type = 3
        print(f"draw box {rect}")
        x, y = rect[0], rect[1]
        cv2.rectangle(self.image, (x, y), (rect[2], rect[3]), outline, line_type)
        return

    def draw_text(
            self: Camera,
            text: str,
            location: Tuple[int, int],
            color: Optional[Tuple[int, int, int]]
    ) -> None:
        if self.image is None:
            return
        font_color = color or self._default_color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        line_type = 2
        x, y = location
        x1 = x + 10 if x < 10 else x - 10
        y1 = y + 20 if y < 20 else y - 20
        cv2.putText(self.image, text, (x1, y1), font, font_scale, font_color, line_type)
        return

    def wait(self, milliseconds: int):
        wait(milliseconds)

    @property
    def dims(self):
        return self._dims


class CvCamera(Camera):
    def __init__(
            self: CvCamera,
            media: Optional[str],
            width: int = 640,
            height: int = 480,
            hflip: bool = False,
            vflip: bool = False,
            threshold: float = 0.25,
            fontsize: int = 10,
            fastforward: int = 0
    ) -> None:
        print(f"CvCamera {media}")
        self.media = media
        self.window = None
        # self.buffer = None
        self.image = None
        self.is_image = False
        if media is None:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            fastforward = 1
        else:
            self.type = Path(media).suffix
            self.is_image = self.type in ['.jpg', '.png']
            self.cam = cv2.VideoCapture(media)

        # adjust aspect ratio
        height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"w {width}, h {height}")
        super().__init__(
            width=width, height=height,
            threshold=threshold, fontsize=fontsize, fastforward=fastforward
        )
        # set flipcode
        if hflip:
            if vflip:
                self.flipcode = -1
            else:
                self.flipcode = 1
        elif vflip:
            self.flipcode = 0
        else:
            self.flipcode = None
        return

    def start(self: CvCamera) -> None:
        self.window = 'Object Detection'
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window, *self._dims)
        # if self.is_image:
        #     _, self.image = self.cam.read()
        return

    def yield_image(self: CvCamera) -> Generator[Image, None]:
        try:
            while True:
                _, image = self.cam.read()
                if image is None:
                    print("eof")
                    return None

                if self.flipcode is not None:
                    image = cv2.flip(image, self.flipcode)
                self.image = image
                yield Image.fromarray(image.copy()[..., ::-1])
        except GeneratorExit:
            print("GeneratorExit")
        return None

    def update(self: CvCamera) -> None:
        if self.image is None:
            return
        print("update")
        cv2.imshow(self.window, self.image)
        wait(1000 // FRAME_PER_SECOND)
        return

    def stop(self: CvCamera) -> None:
        cv2.destroyAllWindows()
        self.cam.release()
        return


def get_camera(
        media: Optional[str],
        height: int,
        width: int,
        hflip: bool,
        vflip: bool,
        threshold: float,
        fontsize: int,
        fastforward: int
) -> Camera:
    return CvCamera(
        media=media,
        width=width, height=height,
        hflip=hflip, vflip=vflip,
        threshold=threshold, fontsize=fontsize
    )


def wait(milliseconds: int):
    key = cv2.waitKey(milliseconds)
    if key == 99:
        raise KeyboardInterrupt


if __name__ == "__main__":
    media = "images/pets.jpg"
    text = "Hello World"
    print("camera self test")
    cam = CvCamera(media, threshold=0.25, fontsize=10)
    cam.start()
    for im in cam.yield_image():
        cam.draw_text("Hello world", (100, 100), (0, 0, 0, 0))
        cam.draw_box((50, 30, 200, 300), (0, 0, 0, 0))
        cam.update()
        wait(1000)

    wait(10000)
    cam.stop()
