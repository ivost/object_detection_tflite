#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import io
import os
import platform
import time
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional

import cv2
import matplotlib.cm as cm
import numpy as np

# from PIL import Image, ImageDraw, ImageFont, ImageColor

if platform.system() == 'RPi':  # RaspberryPi
    import picamera

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
        # self.buffer = Image.new('RGBA', self._dims)
        self.buffer = None
        self.overlay = None
        # self._draw = ImageDraw.Draw(self.buffer)
        self._default_color = (0xff, 0xff, 0xff, 0xff)
        # self._font = ImageFont.truetype("Ubuntu-M.ttf", 40)
        # self._font = ImageFont.truetype(font='TakaoGothic.ttf', size=fontsize)

        self._threshold = threshold
        self._fontsize = fontsize
        self._fastforward = fastforward
        return

    def clear(self: Camera) -> None:
        print("clear")
        # self._draw.rectangle(
        #     (0, 0) + self._dims,
        #     fill=(0, 0, 0, 0x00)
        # )
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
            text, location=(5, 5), color=None
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
        color = tuple(np.array(np.array(cm.jet((prob - self._threshold) / (1.0 - self._threshold))) * 255, dtype=np.uint8).tolist())
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
            self._draw_text(
                name, location=(xoff, yoff), color=color
            )
            yoff += self._fontsize
        self._draw_text(
            '%.3f' % prob, location=(xoff, yoff), color=color
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
        print("draw text", text)
        # self._draw.text(location, text, fill=color,
        #                 stroke_width=2,  font=self._font)
        #
        # self.buffer.show()
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
        # self.overlay = self.cam.add_overlay(
        #     self.buffer.tobytes(),
        #     format='rgba', layer=3, size=self._dims
        # )
        #
        # self.overlay.update(self.buffer.tobytes())
        return


class PiCamera(Camera):
    def __init__(
            self: PiCamera,
            width: int,
            height: int,
            hflip: bool,
            vflip: bool,
            threshold: float,
            fontsize: int
    ) -> None:
        super().__init__(
            width=width, height=height,
            threshold=threshold, fontsize=fontsize
        )
        self.cam = picamera.PiCamera(
            resolution=(width, height),
            framerate=FRAME_PER_SECOND
        )
        self.cam.hflip = hflip
        self.cam.vflip = vflip
        return

    def start(self: PiCamera) -> None:
        self.cam.start_preview()
        return

    # def yield_image(self: PiCamera) -> Generator[Image, None]:
    #     self._stream = io.BytesIO()
    #     for _ in self.cam.capture_continuous(
    #             self._stream,
    #             format='jpeg',
    #             use_video_port=True
    #     ):
    #         self._stream.seek(0)
    #         image = Image.open(self._stream).convert('RGB')
    #         yield image
    #     return

    def update(self: PiCamera) -> None:
        super().update()
        self._stream.seek(0)
        self._stream.truncate()
        return

    def stop(self: PiCamera) -> None:
        self.cam.stop_preview()
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
            _, self.buffer = self.cam.read()
        return

    def yield_image(self: CvCamera) -> Generator[np.ndarray, None]:
        while True:
            _, self.buffer = self.cam.read()
            if self.buffer is None:
                time.sleep(1)
                continue
            if self.flipcode is not None:
                frame = cv2.flip(self.buffer, self.flipcode)
            yield self.buffer
            # self.image = image
            # yield Image.fromarray(image.copy()[..., ::-1])
        return

    def update(self: CvCamera) -> None:
        if self.buffer is None:
            return

        # overlay = np.array(self.buffer, dtype=np.uint8)
        # overlay = cv2.cvtColor(self.buffer, cv2.COLOR_RGBA2BGRA)
        # image = cv2.addWeighted(
        #     cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
        #     overlay, 0.5, 2.2
        # )

        cv2.imshow(self.window, self.buffer)
        key = cv2.waitKey(1000 // FRAME_PER_SECOND)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: CvCamera) -> None:
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        self.cam.release()
        return

# class PlCamera(Camera):
#     def __init__(
#             self: PlCamera,
#             media: Optional[str],
#             threshold: float,
#             fontsize: int
#     ) -> None:
#         print(f"==== PlCamera {media}")
#         self.image = Image.open(media)
#         width, height = self.image.size
#         print(f"format: {self.image.format}, size: {self.image.size}, mode: {self.image.mode}")
#         self.image = self.image.convert("RGBA")
#
#         super().__init__(width, height, threshold, fontsize)
#
#         return
#
#     def start(self: PlCamera) -> None:
#         self.window = 'Object Detection'
#         cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
#         cv2.resizeWindow(self.window, *self._dims)
#         return
#
#     def yield_image(self: PlCamera) -> Image:
#         return Image.fromarray(self.image.copy()[..., ::-1])
#
#     def update(self: PlCamera) -> None:
#         overlay = np.array(self.buffer, dtype=np.uint8)
#         overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
#         image = cv2.addWeighted(
#             cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
#             overlay, 0.5, 2.2
#         )
#         cv2.imshow(self.window, image)
#         key = cv2.waitKey(0)
#         if key == 99:
#             raise KeyboardInterrupt
#         return
#
#     def stop(self: PlCamera) -> None:
#         cv2.destroyAllWindows()
#         return


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
    if platform.system() == 'RPi':  # RaspberryPi
        return PiCamera(
            width=width, height=height,
            hflip=hflip, vflip=vflip,
            threshold=threshold, fontsize=fontsize
        )
    return CvCamera(
        media=media,
        width=width, height=height,
        hflip=hflip, vflip=vflip,
        threshold=threshold, fontsize=fontsize
    )

# def image_draw_test():
#     im = Image.open("images/parrot.jpg")
#     # im = im.convert("RGBA")
#     print(f"format: {im.format}, size: {im.size}, mode: {im.mode}")
#     c1 = ImageColor.getrgb("white")
#     c2 = ImageColor.getrgb("yellow")
#     draw = ImageDraw.Draw(im)
#     draw.line((0, 0) + im.size, fill=c1, width=2)
#     draw.line((0, im.size[1], im.size[0], 0), fill=c2, width=4)
#
#     # make a blank image for the text, initialized to transparent text color
#     # txt = Image.new("RGBA", im.size, (255, 255, 255, 0))
#
#     # get a font
#     # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
#     # fnt = ImageFont.truetype("FreeMono.ttf", 20)
#     # fnt = ImageFont.truetype("Ubuntu-M.ttf", 40)
#     fnt = ImageFont.truetype("TakaoGothic.ttf", 40)
#     # get a drawing context
#     # d = ImageDraw.Draw(txt)
#     d = ImageDraw.Draw(im)
#
#     # draw text, half opacity
#     d.text((10, 10), "Hello", font=fnt, fill=c1)
#     # draw text, full opacity
#     d.text((10, 60), "World", font=fnt, fill=ImageColor.getrgb("black"))
#
#     # out = Image.alpha_composite(im, txt)
#     # out.show()
#     im.show()
#
#     return

def capture_test(media):
    cap = cv2.VideoCapture(media)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"h {h}, w {w}")

    ok, frame = cap.read()
    if not ok:
        print("ERROR")
        return
    h, w, c = frame.shape
    print(f"h {h}, w {w}")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10000) & 0xFF
    cv2.destroyAllWindows()


if __name__ == "__main__":
    media = "images/pets.jpg"
#    capture_test(media)
#    draw_test()

    print("camera self test")
    cam = CvCamera(media, threshold=0.25, fontsize=16)
    cam.start()
    cam.draw_time(1000)
    cam.update()
    cam.stop()
