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

"""
pass