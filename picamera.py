"""

if platform.system() == 'RPi':  # RaspberryPi
    import picamera


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


    if platform.system() == 'RPi':  # RaspberryPi
        return PiCamera(
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

"""
pass