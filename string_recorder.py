"""shamelessly stolen from
https://github.com/kiyukuta/string_recorder/blob/master/string_recorder/string_recorder.py"""

import json
import os
import re

import imageio
import numpy
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import PIL.ImageFont

# get monokai colors from https://marketplace.visualstudio.com/items?itemName=SuperPaintman.monokai-extended#colors
colors = [
    "#75715E",  # gray
    "#F92672",  # red
    "#A6E22E",  # green
    "#E6DB74",  # yellow
    "#66D9EF",  # blue
    "magenta",
    "cyan",
    "white",
    "crimson",
]


def get_font():
    font_name = "monospace.medium.ttf"
    font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
    if not os.path.exists(font_path):
        raise RuntimeError(f"download font first and put it in {str(font_path)}")
    return font_path


class StringRecorder(object):
    def __init__(self, font=None, max_frames=100000):
        self.max_frames = max_frames
        if font is None:
            font = get_font()
        self.font = PIL.ImageFont.truetype(font, size=80)
        self.height = -1
        self.width = -1

        self.tmpdraw = PIL.ImageDraw.Draw(PIL.Image.new("RGB", (1, 1)))
        self._images = []
        self._sizes = []
        self._spacing = 0

        self._step = 0

        self.reg1 = re.compile("((?:\u001b\[[0-9;]+?m){0,2}.+?(?:\u001b\[0m){0,2})")
        self.reg2 = re.compile("((?:\u001b\[[0-9;]+?m){0,2})(.+?)(?:\u001b\[0m){0,2}")
        self.bg_reg = re.compile("\u001b\[(4[0-9;]+?)m")
        self.fg_reg = re.compile("\u001b\[(3[0-9;]+?)m")

    def reset(self):
        self.height = -1
        self.width = -1
        self._images = []
        self._sizes = []
        self._step = 0

    def render(self, frame):

        splitted = [[w for w in self.reg1.split(l) if w != ""] for l in frame.split("\n")]
        parsed = [[self.reg2.findall(c) for c in l] for l in splitted]

        frame = "\n".join(["".join(c[0][1] for c in l) for l in parsed])

        d = {}
        for y, row in enumerate(parsed):
            for x, col in enumerate(row):
                if col[0][0] == "":
                    continue
                d[(y, x)] = col[0][0]

        size = self.tmpdraw.textsize(frame, font=self.font, spacing=self._spacing)
        image = PIL.Image.new("RGB", size, "#272822")
        draw = PIL.ImageDraw.Draw(image, mode="RGB")

        cw, ch = self.tmpdraw.textsize("A", font=self.font, spacing=self._spacing)

        for k, v in d.items():
            y, x = k
            background = self.bg_reg.findall(v)

            if background != []:
                assert background[0][0] == "4"
                color = PIL.ImageColor.getrgb(colors[int(background[0][1])])
                draw.rectangle((cw * x, ch * y, cw * (x + 1), ch * (y + 1)), fill=color)

        draw.text((0, 0), frame, font=self.font, fill="#f8f8f2", spacing=self._spacing)

        for k, v in d.items():
            y, x = k
            foreground = self.fg_reg.findall(v)

            if foreground != []:
                assert foreground[0][0] == "3"
                color = PIL.ImageColor.getrgb(colors[int(foreground[0][1])])
                char = parsed[y][x][0][1]

                draw.text((cw * x, ch * y), char, font=self.font, fill=color, spacing=self._spacing)

        return image, size

    def record_frame(self, frame):
        assert type(frame) == str

        image, (width, height) = self.render(frame=frame)
        self._images.append(image)
        if self.width < width:
            self.width = width
        if self.height < height:
            self.height = height
        self._step += 1

    def make_video(self, save_path, fps=2.5):
        if not save_path.endswith(".mp4"):
            save_path += ".mp4"
        images = []
        for img in self._images:
            image = PIL.Image.new("RGB", (self.width, self.height), "#f8f8f2")
            image.paste(img, box=(0, 0))
            image = image.resize((688, 784))  # to avoid warning
            images.append(numpy.asarray(image))

        imageio.mimsave(save_path, images, fps=fps)
        self.reset()

    @property
    def step(self):
        return self._step
