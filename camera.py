#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional

import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

FRAME_PER_SECOND = 30
FONT = "Roboto-Black.ttf"


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
        self.buffer = Image.new('RGBA', self._dims)
        self.overlay = None
        self._draw = ImageDraw.Draw(self.buffer)
        self._default_color = (0xff, 0xff, 0xff, 0xff)
        self._font = ImageFont.truetype(FONT, fontsize)
        self._threshold = threshold
        self._fontsize = fontsize
        self._fastforward = fastforward
        return

    def clear(self: Camera) -> None:
        print("clear")
        self._draw.rectangle(
            (0, 0) + self._dims,
            fill=(0, 0, 0, 0x00)
        )
        return

    def draw_objects(self: Camera, objects: List) -> None:
        for obj in objects:
            self._draw_object(obj)
        return

    def draw_time(self: Camera, elapsed_ms: float) -> None:
        text = 'Elapsed Time: %.1f[ms]' % elapsed_ms
        if self._fastforward > 1:
            text += ' (speed x%d)' % self._fastforward
        self._draw_text(
            text, location=(10, 10), color=None
        )
        return

    def draw_count(self: Camera, count: int) -> None:
        self._draw_text(
            'Detected Objects: %d' % count,
            location=(5, 5 + self._fontsize), color=None
        )
        return

    def _draw_object(self: Camera, object: Dict) -> None:
        prob = object['prob']
        color = tuple(np.array(np.array(cm.jet((prob - self._threshold) / (1.0 - self._threshold))) * 255,
                               dtype=np.uint8).tolist())
        color = (0, 80, 100, 0)
        print(f"draw prob {prob}, font {self._font.path}, color {color}")
        self._draw_box(
            rect=object['bbox'], color=color
        )
        name = object.get('name')
        xoff = object['bbox'][0] + 5
        yoff = object['bbox'][1] + 5

        if name is not None:
            print(xoff, yoff, name)
            self._draw_text(name, location=(xoff, yoff), color=color)
            yoff += self._fontsize
        self._draw_text(f"{prob:%.3f'}", location=(xoff, yoff), color=color
                        )
        return

    def _draw_box(
            self: Camera,
            rect: Tuple[int, int, int, int],
            color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        outline = color or self._default_color
        print("draw box")
        # self._draw.rectangle(rect, fill=None, outline=outline)
        return

    def _draw_text(
            self: Camera,
            text: str,
            location: Tuple[int, int],
            color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        color = color or self._default_color
        print("draw text", text, "color", color)
        self._draw.text(location, text, fill=color,
                        stroke_width=4, font=self._font)

        self.buffer.show()
        return

    def update(self: Camera) -> None:
        print("update")
        # self.buffer.show()
        if self.overlay is not None:
            print("remove_overlay")
            self.cam.remove_overlay(self.overlay)

        if self.buffer is None:
            print("no buffer")
            return
        print("todo update")
        self.overlay = self.cam.add_overlay(
            self.buffer.tobytes(),
            format='rgba', layer=3, size=self._dims
        )

        self.overlay.update(self.buffer.tobytes())
        return


class CvCamera(Camera):
    def __init__(
            self: CvCamera,
            media: Optional[str],
            width: int = 640,
            height: int = 480,
            hflip: bool = False,
            vflip: bool = False,
            threshold: float = 0.25,
            fontsize: int = 20,
            fastforward: int = 0
    ) -> None:
        print(f"==== CvCamera {media}")
        self.media = media
        self.window = None
        self.buffer = None
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
        if self.is_image:
            _, self.image = self.cam.read()
        return

    def yield_image(self: CvCamera) -> Generator[Image, None]:
        while True:
            _, image = self._camera.read()
            if image is None:
                time.sleep(1)
                continue
            if self.flipcode is not None:
                image = cv2.flip(image, self.flipcode)
            self.image = image
            yield Image.fromarray(image.copy()[..., ::-1])
        return

    def update(self: CvCamera) -> None:
        if self.buffer is None:
            return

        if self.image is None:
            return

        overlay = np.array(self.buffer, dtype=np.uint8)
        # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        self.image = cv2.addWeighted(cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA),
                                     0.5, overlay, 0.5, 2.2)
        cv2.imshow(self.window, self.image)
        key = cv2.waitKey(1000 // FRAME_PER_SECOND)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: CvCamera) -> None:
        cv2.waitKey(2000)
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


if __name__ == "__main__":
    media = "images/pets.jpg"
    cap = cv2.VideoCapture(media)
    ok, image = cap.read()
    if not ok:
        print("error")
        exit(1)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"h {h}, w {w}")
    # im = Image.open(media)
    # im = im.convert("RGBA")
    # print(f"format: {im.format}, size: {im.size}, mode: {im.mode}")
    dims = (w, h)
    # c1 = ImageColor.getrgb("black")
    # c2 = ImageColor.getrgb("red")
    # # make a blank image for the text, initialized to transparent text color
    # buffer = Image.new("RGBA", dims, (255, 255, 255, 0))
    # fnt = ImageFont.truetype(FONT, 40)
    # # get a drawing context
    # d = ImageDraw.Draw(buffer)
    # # draw text, half opacity
    # d.text((10, 10), "Hello", font=fnt, fill=c1)
    # # draw text, full opacity
    # d.text((10, 60), "World", font=fnt, fill=c2)
    # overlay = np.array(buffer, dtype=np.uint8)
    # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
    # im = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # # image = cv2.addWeighted(im, 0.5, overlay, 0.5, 0)
    # image = cv2.addWeighted(im, 0.9, overlay, 0.1, 0)
    # out = Image.alpha_composite(im, txt)
    # out.show()
    text = "Hello World"
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = color
    line_type = 2
    box = (10, 10, 640, 480)
    x, y = box[0], box[1]
    x1 = x + 10 if x < 10 else x - 10
    y1 = y + 20 if y < 20 else y - 20
    cv2.rectangle(image, (x, y), (box[2], box[3]), color, line_type)
    cv2.putText(image, text, (x1, y1), font, font_scale, font_color, line_type)

    cv2.imshow("Frame", image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    '''
    dims = (frame.shape[0], frame.shape[1])   
    buffer = Image.new('RGBA', dims)
    '''

    # print("camera self test")
    # cam = CvCamera(media, threshold=0.25, fontsize=10)
    # cam.start()
    # cam.draw_time(1000)
    # cam.update()
    # cam.stop()
