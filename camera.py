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
        self._buffer = Image.new('RGBA', self._dims)
        self._overlay = None
        self._draw = ImageDraw.Draw(self._buffer)
        self._default_color = (0xff, 0xff, 0xff, 0xff)
        # self._font = ImageFont.truetype("Ubuntu-M.ttf", 40)
        self._font = ImageFont.truetype(font='TakaoGothic.ttf', size=fontsize)

        self._threshold = threshold
        self._fontsize = fontsize
        self._fastforward = fastforward
        return

    def clear(self: Camera) -> None:
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
        self._draw.rectangle(rect, fill=None, outline=outline)
        return

    def _draw_text(
            self: Camera,
            text: str,
            location: Tuple[int, int],
            color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        color = color or self._default_color
        print(text)
        self._draw.text(location, text, fill=color,
                        stroke_width=2,  font=self._font)

        # self._buffer.show()
        return

    def update(self: Camera) -> None:
        print("update")
        # self._buffer.show()

        if self._overlay is not None:
            self._camera.remove_overlay(self._overlay)

        if self._buffer is None:
            return

        self._overlay = self._camera.add_overlay(
            self._buffer.tobytes(),
            format='rgba', layer=3, size=self._dims
        )

        self._overlay.update(self._buffer.tobytes())

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
        self._camera = picamera.PiCamera(
            resolution=(width, height),
            framerate=FRAME_PER_SECOND
        )
        self._camera.hflip = hflip
        self._camera.vflip = vflip
        return

    def start(self: PiCamera) -> None:
        self._camera.start_preview()
        return

    def yield_image(self: PiCamera) -> Generator[Image, None]:
        self._stream = io.BytesIO()
        for _ in self._camera.capture_continuous(
                self._stream,
                format='jpeg',
                use_video_port=True
        ):
            self._stream.seek(0)
            image = Image.open(self._stream).convert('RGB')
            yield image
        return

    def update(self: PiCamera) -> None:
        super().update()
        self._stream.seek(0)
        self._stream.truncate()
        return

    def stop(self: PiCamera) -> None:
        self._camera.stop_preview()
        return


class CvCamera(Camera):
    def __init__(
            self: CvCamera,
            media: Optional[str],
            width: int,
            height: int,
            hflip: bool,
            vflip: bool,
            threshold: float,
            fontsize: int,
            fastforward: int
    ) -> None:
        print(f"==== CvCamera {media}")
        if media is None:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            fastforward = 1
        else:
            self._camera = cv2.VideoCapture(media)
        # adjust aspect ratio
        height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        overlay = np.array(self._buffer, dtype=np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        image = cv2.addWeighted(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
            overlay, 0.5, 2.2
        )
        cv2.imshow(self.window, image)
        key = cv2.waitKey(1000 // FRAME_PER_SECOND)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: CvCamera) -> None:
        cv2.destroyAllWindows()
        self._camera.release()
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
#         overlay = np.array(self._buffer, dtype=np.uint8)
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
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        threshold: float,
        fontsize: int,
        fastforward: int
) -> Camera:

    assert media is not None

    if not os.path.exists(media):
        raise ValueError(f'{media} not found')

    # if Path(media).suffix in ['.jpg', '.png']:
    #     return PlCamera(
    #         media=media, threshold=threshold, fontsize=fontsize
    #     )

    return CvCamera(
        media=media,
        width=width, height=height,
        hflip=hflip, vflip=vflip,
        threshold=threshold, fontsize=fontsize,
        fastforward=fastforward
    )

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


# def draw_test():
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


if __name__ == "__main__":
#    draw_test()

    # # test
    # print("camera self test")
    # cam = PlCamera("images/parrot.jpg", threshold=0.25, fontsize=16)
    # cam.start()
    # cam.clear()
    # cam.draw_time(1000)
    # cam.stop()
