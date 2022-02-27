from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import getopt
import glob
import math
import ntpath
import re
import sys

from PIL import Image, ImageEnhance, ImageOps, ImageSequence, ImageFilter, ImageDraw

usage_string = '''Usage: imgop operation,... [options] file ...
Options:
  -o output            Output file name (requires one input file) 
  -b background        Background color name or RGB/RGBA hex code with #, 'none' for transparent
  -q quality           Output quality % (default 95)
  -f frame             Frame index if input file is an animated GIF
  -c cutoff            For ac op: cut-off % (default 5)
  -r crop rect         For kr op: rectangle x1/y1/x2/y2
  -x offset            For sz op: x offset in pixels, negative aligns at the right
  -y offset            For sz op: y offset in pixels, negative aligns at the bottom
  -w width             For rsz op: width in pixels
  -h height            For rsz op: height in pixels
  --overwrite          Do not append postfixes, overwrites input files if file extension is the same
  --strip_icc_profile  For conversion ops: remove icc_profile from image
  --halign 0/1/2       For sz and sp2 op: horizontal alignment 0=left, 1=center, 2=right 
  --valign 0/1/2       For sz and sp2 op: vertical alignment 0=top, 1=center, 2=bottom

To run multiple operations use a comma separated list without spaces.
Operations that require multiple input files should be run standalone.
Otherwise the same operations will be done on each file specified (wildcards are allowed).
Result is written to new file with operations as postfix unless --overwrite is specified.

To show a list of all operations: imgop list
'''


def rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None):
    upper_left_point = xy[0]
    bottom_right_point = xy[1]
    draw.rectangle(
        [
            (upper_left_point[0], upper_left_point[1] + corner_radius),
            (bottom_right_point[0], bottom_right_point[1] - corner_radius)
        ],
        fill=fill,
        outline=outline
    )
    draw.rectangle(
        [
            (upper_left_point[0] + corner_radius, upper_left_point[1]),
            (bottom_right_point[0] - corner_radius, bottom_right_point[1])
        ],
        fill=fill,
        outline=outline
    )
    draw.pieslice(
        [
            upper_left_point,
            (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)
        ],
        180,
        270,
        fill=fill,
        outline=outline
    )
    draw.pieslice(
        [
            (bottom_right_point[0] - corner_radius * 2, bottom_right_point[1] - corner_radius * 2),
            bottom_right_point
        ],
        0,
        90,
        fill=fill,
        outline=outline
    )
    draw.pieslice(
        [
            (upper_left_point[0], bottom_right_point[1] - corner_radius * 2),
            (upper_left_point[0] + corner_radius * 2, bottom_right_point[1])
        ],
        90,
        180,
        fill=fill,
        outline=outline
    )
    draw.pieslice(
        [
            (bottom_right_point[0] - corner_radius * 2, upper_left_point[1]),
            (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)
        ],
        270,
        360,
        fill=fill,
        outline=outline
    )


def mask_circle_transparent(im, offset=0, blur_radius=0):
    offset = blur_radius * 2 + offset
    mask = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(mask)
    xy = (offset, offset, im.size[0] - offset, im.size[1] - offset)
    draw.ellipse(xy, fill=255)
    if blur_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    result = im.copy()
    result.putalpha(mask)
    return result


def mask_rounded_rectangle_transparent(im, offset=0, corner_radius=0):
    mask = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(mask)
    xy = ((offset, offset), (im.size[0] - 2 * offset, im.size[1] - 2 * offset))
    rounded_rectangle(draw, xy, corner_radius, fill=255)
    result = im.copy()
    result.putalpha(mask)
    return result


def _round_up_to_pow(n, p=2):
    return int(round(pow(p, math.ceil(math.log(n, p)))))


def _align_offset(src_size, dst_size, align):
    if align == 0:
        return 0
    if align == 2:
        return dst_size - src_size
    return (dst_size - src_size) // 2


def _translate_offset(offset, src_size, dst_size):
    if offset >= 0:
        return offset
    return dst_size - src_size + offset + 1


def tile(ims, m, n, bgcolor):
    if not isinstance(ims, list):
        size = ims.size
        ims = [ims] * m * n
    else:
        ims = ims[:m * n]
        # take tallest or widest
        size = max((im.size for im in ims), key=lambda s: s[0 if m < n else 1])
    w, h = size
    fw, fh = float(w), float(h)
    if w > h and m > n or w < h and m < n:
        m, n = n, m

    new_ims = []
    for i, im in enumerate(ims):
        if im.size != size:
            w2, h2 = im.size
            if w > h and w2 < h2 or w < h and w2 > h2:
                im = im.transpose(Image.ROTATE_90)
                w2, h2 = im.size
            im = resize(im, max(fw // w2, fh // h2))
        new_ims.append(im)

    nw = 0
    nh = 0
    for im in new_ims[0:m]:
        nw += im.size[0]
    for im in new_ims[0:n * m:m]:
        nh += im.size[1]

    nsize = nw, nh
    im2 = Image.new(ims[0].mode, nsize, bgcolor)
    osx = 0
    osy = 0
    for i, im in enumerate(new_ims):
        if i % m == 0:
            osx = 0
        im2.paste(im, (osx, osy))
        osx += im.size[0]
        if i % m == m - 1:
            osy += im.size[1]
    return im2


def tile3(im):
    im2 = resize(im, 0.5)
    w, h = im.size
    w2, h2 = im2.size
    nw = w + w2
    nsize = nw, h
    im3 = Image.new(im.mode, nsize, None)
    im3.paste(im, (0, 0))
    im3.paste(im2, (w, 0))
    im3.paste(im2, (w, h2))
    return im3


def index(ims, bgcolor):
    count = len(ims)
    if count <= 1:
        return None
    m = int(math.sqrt(count / 1.5))
    n = int((count + (m - 1)) / m)
    if count > 50:
        ims = [resize(im, 2.0 / (m + n)) for im in ims]
    return tile(ims, m, n, bgcolor)


def aspect_ratio(im, ar, bgcolor, force=True, allow_swap=True):
    w, h = im.size
    if allow_swap and (w > h and ar < 1.0 or w < h and ar > 1.0):
        ar = 1.0 / ar
    nw = int(h * ar + 0.5)
    if nw >= w:
        nh = h
    else:
        nh = int(w / ar + 0.5)
        nw = w
    wd = nw - w
    hd = nh - h
    if wd < 10 and hd < 10:
        return force and im
    nsize = nw, nh
    im2 = Image.new(im.mode, nsize, bgcolor)
    im2.paste(im, (wd // 2, hd // 2))
    return im2


def a23f(im, settings):
    return aspect_ratio(im, 3.0 / 2.0, settings.bgcolor, force=True)


def add_border(im, r, bgcolor):
    r2 = 1.0 + 2 * r
    w, h = im.size
    nw = int(w * r2 + 0.5)
    nh = int(h * r2 + 0.5)
    nsize = nw, nh
    im2 = Image.new(im.mode, nsize, bgcolor)
    im2.paste(im, ((nw - w) // 2, (nh - h) // 2))
    return im2


def convert(im, settings, mode='RGB'):
    if not settings.overwrite and im.format == settings.format:
        print("Image already in format '%s'" % im.format)
        return None
    if im.format == 'GIF' and settings.frame > 0:
        im = ImageSequence.Iterator(im)[settings.frame]
    if settings.strip_icc_profile:
        im.info.pop('icc_profile', None)
    return im.convert(mode)


def resize(im, fw, fh=0.0):
    if fw == 0.0:
        return im
    if fh == 0.0:
        fh = fw
    w, h = im.size
    w = int(fw * w)
    h = int(fh * h)
    return im.resize((w, h), Image.ANTIALIAS)


def resize_pow2(im, scale_width, bgcolor):
    w, h = im.size
    if scale_width:
        nh = _round_up_to_pow(h)
        nw = int((w * nh + h // 2) / h)
    else:
        nw = _round_up_to_pow(w)
        nh = int((h * nw + w // 2) / w)
    im2 = im if nw == w and nh == h else im.resize((nw, nh), Image.ANTIALIAS)
    im3 = canvas_size_pow2(im2, bgcolor)
    if im3 is None:
        return None if im2 is im else im2
    return im3


def canvas_size(im, bgcolor, width, height, x=0, y=0, halign=0, valign=0):
    if not width and not height:
        return None
    if not width:
        width = height
    elif not height:
        height = width
    w, h = im.size
    if w == width and h == height:
        return None
    im2 = Image.new(im.mode, (width, height), bgcolor)
    nx = x + _align_offset(w, width, halign)
    ny = y + _align_offset(h, height, valign)
    nx = _translate_offset(nx, w, width)
    ny = _translate_offset(ny, h, height)
    im2.paste(im, (nx, ny))
    return im2


def canvas_size_pow2(im, bgcolor, halign=0, valign=0):
    w, h = im.size
    nw = _round_up_to_pow(w)
    nh = _round_up_to_pow(h)
    return canvas_size(im, bgcolor, nw, nh, halign=halign, valign=valign)


# def contrast(im, f):
#     return ImageEnhance.Contrast(im).enhance(f)


def brightness(im, f):
    return ImageEnhance.Brightness(im).enhance(f)


def box_blur(im, pixels):
    return im.filter(ImageFilter.BoxBlur(pixels))


def gaussian_blur(im, radius):
    return im.filter(ImageFilter.GaussianBlur(radius))


def sharpen(im, f):
    return ImageEnhance.Sharpness(im).enhance(f)


def enhance_color(im, f):
    enhancer = ImageEnhance.Color(im)
    return enhancer.enhance(f)


def x2(im, settings):
    return tile(im, 2, 1, settings.bgcolor)


def x3(im, settings):
    im = tile(im, 3, 1, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x4(im, settings):
    im = tile(im, 2, 2, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x5(im, settings):
    im = tile(im, 5, 1, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x6(im, settings):
    im = tile(im, 3, 2, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x8(im, settings):
    im = tile(im, 4, 2, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x9(im, settings):
    im = tile(im, 3, 3, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def x12(im, settings):
    im = tile(im, 4, 3, settings.bgcolor)
    if im.size[0] > 2000 or im.size[1] > 2000:
        im = resize(im, 0.5)
    return im


def t2(ims, settings):
    return x2(ims, settings)


def t3(ims, settings):
    return x3(ims, settings)


def t4(ims, settings):
    return x4(ims, settings)


def t6(ims, settings):
    return x6(ims, settings)


def t8(ims, settings):
    return x8(ims, settings)


def t9(ims, settings):
    return x9(ims, settings)


def t12(ims, settings):
    return x12(ims, settings)


def m2(im, settings):
    mim = ImageOps.mirror(im)
    ims = [mim, im]
    return x2(ims, settings)


def m4(im, settings):
    mim = ImageOps.mirror(im)
    ims = [mim, im, mim, im]
    return x4(ims, settings)


def idx(ims, settings):
    return index(ims, settings.bgcolor)


# noinspection PyUnusedLocal
def bw(im, settings):
    # enhancer = ImageEnhance.Color(im)
    # return enhancer.enhance(0.0)
    return ImageOps.grayscale(im)


# noinspection PyUnusedLocal
def en5(im, settings):
    return enhance_color(im, 1.05)


# noinspection PyUnusedLocal
def en10(im, settings):
    return enhance_color(im, 1.1)


# noinspection PyUnusedLocal
def en20(im, settings):
    return enhance_color(im, 1.2)


def ac(im, settings):
    return ImageOps.autocontrast(im, settings.cutoff)


def a13(im, settings):
    return aspect_ratio(im, 13.0 / 10.0, settings.bgcolor, force=False)


def a23(im, settings):
    return aspect_ratio(im, 3.0 / 2.0, settings.bgcolor, force=False)


def a23s(im, settings):
    return aspect_ratio(im, 3.0 / 2.0, settings.bgcolor, force=False, allow_swap=False)


def a23p(im, settings):
    return aspect_ratio(im, 2.0 / 3.0, settings.bgcolor, force=False, allow_swap=False)


def a34(im, settings):
    return aspect_ratio(im, 4.0 / 3.0, settings.bgcolor, force=False)


def a34s(im, settings):
    return aspect_ratio(im, 4.0 / 3.0, settings.bgcolor, force=False, allow_swap=False)


def a34p(im, settings):
    return aspect_ratio(im, 3.0 / 4.0, settings.bgcolor, force=False, allow_swap=False)


def a45(im, settings):
    return aspect_ratio(im, 5.0 / 4.0, settings.bgcolor, force=False)


def a45s(im, settings):
    return aspect_ratio(im, 5.0 / 4.0, settings.bgcolor, force=False, allow_swap=False)


def a45p(im, settings):
    return aspect_ratio(im, 4.0 / 5.0, settings.bgcolor, force=False, allow_swap=False)


def a4(im, settings):
    return aspect_ratio(im, math.sqrt(2.0), settings.bgcolor, force=False)


def a4s(im, settings):
    return aspect_ratio(im, math.sqrt(2.0), settings.bgcolor, force=False, allow_swap=False)


def a4p(im, settings):
    return aspect_ratio(im, 1.0 / math.sqrt(2.0), settings.bgcolor, force=False, allow_swap=False)


def ap2(im, settings):
    return resize_pow2(im, False, settings.bgcolor)


def e5(im, settings):
    return add_border(im, 0.05, settings.bgcolor)


def e10(im, settings):
    return add_border(im, 0.1, settings.bgcolor)


def e20(im, settings):
    return add_border(im, 0.2, settings.bgcolor)


def e30(im, settings):
    return add_border(im, 0.3, settings.bgcolor)


def sz(im, settings):
    return canvas_size(im, settings.bgcolor, settings.width, settings.height,
                       x=settings.x, y=settings.y, halign=settings.halign, valign=settings.valign)


def sp2(im, settings):
    return canvas_size_pow2(im, settings.bgcolor, halign=settings.halign, valign=settings.valign)


def cjpg(im, settings):
    settings.format = 'JPEG'
    return convert(im, settings)


def cpng(im, settings):
    settings.format = 'PNG'
    return convert(im, settings, 'RGBA')


def ctga(im, settings):
    settings.format = 'TGA'
    im2 = convert(im, settings, 'RGBA')
    return im2


def rsz(im, settings):
    width = settings.width
    height = settings.height
    if not width and not height:
        return None
    w, h = im.size
    if not width:
        fw = float(height) / float(h)
        width = int(fw * w)
    elif not height:
        fh = float(width) / float(w)
        height = int(fh * h)
    return im.resize((width, height), Image.ANTIALIAS)


# noinspection PyUnusedLocal
def r25(im, settings):
    return resize(im, 0.25)


# noinspection PyUnusedLocal
def r50(im, settings):
    return resize(im, 0.5)


# noinspection PyUnusedLocal
def r75(im, settings):
    return resize(im, 0.75)


# noinspection PyUnusedLocal
def r200(im, settings):
    return resize(im, 2.0)


# noinspection PyUnusedLocal
def _hxx(im, f, settings):
    return resize(im, 1.0, f)


def h90(im, settings):
    return _hxx(im, 0.9, settings)


def h95(im, settings):
    return _hxx(im, 0.95, settings)


# noinspection PyUnusedLocal
def _wxx(im, f, settings):
    return resize(im, f, 1.0)


def w90(im, settings):
    return _wxx(im, 0.9, settings)


def w95(im, settings):
    return _wxx(im, 0.95, settings)


# noinspection PyUnusedLocal
def mir(im, settings):
    return ImageOps.mirror(im)


# noinspection PyUnusedLocal
def flp(im, settings):
    return ImageOps.flip(im)


# noinspection PyUnusedLocal
def b5(im, settings):
    return brightness(im, 1.05)


# noinspection PyUnusedLocal
def b10(im, settings):
    return brightness(im, 1.1)


# noinspection PyUnusedLocal
def b20(im, settings):
    return brightness(im, 1.2)


# noinspection PyUnusedLocal
def b50(im, settings):
    return brightness(im, 1.5)


# noinspection PyUnusedLocal
def b100(im, settings):
    return brightness(im, 2.0)


# noinspection PyUnusedLocal
def d5(im, settings):
    return brightness(im, 0.95)


# noinspection PyUnusedLocal
def d10(im, settings):
    return brightness(im, 0.9)


# noinspection PyUnusedLocal
def d20(im, settings):
    return brightness(im, 0.8)


# noinspection PyUnusedLocal
def gb2(im, settings):
    return gaussian_blur(im, 2)


# noinspection PyUnusedLocal
def gb4(im, settings):
    return gaussian_blur(im, 4)


# noinspection PyUnusedLocal
def gb8(im, settings):
    return gaussian_blur(im, 8)


# noinspection PyUnusedLocal
def gb12(im, settings):
    return gaussian_blur(im, 12)


# noinspection PyUnusedLocal
def gb10(im, settings):
    return gaussian_blur(im, 10)


# noinspection PyUnusedLocal
def gb15(im, settings):
    return gaussian_blur(im, 15)


# noinspection PyUnusedLocal
def gb20(im, settings):
    return gaussian_blur(im, 20)


# noinspection PyUnusedLocal
def sh(im, settings):
    return sharpen(im, 4.0)


def kr(im, settings):
    r = settings.crop_rect
    if not r:
        return None
    return im.crop(r)


def u3(im, settings):
    return a23f(tile3(im), settings)


operlist = [
    (x2, 'tile image 2x1 times'),
    (x3, 'tile image 3x1 times'),
    (x4, 'tile image 2x2 times'),
    (x5, 'tile image 5x1 times'),
    (x6, 'tile image 3x2 times'),
    (x8, 'tile image 4x2 times'),
    (x9, 'tile image 3x3 times'),
    (x12, 'tile image 4x3 times'),
    (t2, 'tile 2x1 images, specify 2 input files'),
    (t3, 'tile 3x1 images, specify 3 input files'),
    (t4, 'tile 2x2 images, specify 4 input files'),
    (t6, 'tile 3x2 images, specify 6 input files'),
    (t8, 'tile 4x2 images, specify 8 input files'),
    (t9, 'tile 3x3 images, specify 9 input files'),
    (t12, 'tile 4x3 images, specify 12 input files'),
    (u3, 'tile image 1 + 2x times'),
    (m2, 'mirror and tile image 2x1 times'),
    (m4, 'mirror and tile image 2x2 times'),
    (idx, 'create thumbnail index of all input files'),
    (ac, 'apply automatic contrast, specify cutoff with -c (default 5%)'),
    (bw, 'convert image to black & white'),
    (en5, 'adjust image color balance by 5%'),
    (en10, 'adjust image color balance by 10%'),
    (en20, 'adjust image color balance by 20%'),
    (a13, 'enlarge image to make size 10:13'),
    (a23, 'enlarge image to make size 2:3'),
    (a23p, 'enlarge image to make size 2:3 (force portrait)'),
    (a23s, 'enlarge image to make size 2:3 (force landscape)'),
    (a34, 'enlarge image to make size 3:4'),
    (a34p, 'enlarge image to make size 3:4 (force portrait)'),
    (a34s, 'enlarge image to make size 3:4 (force landscape)'),
    (a45, 'enlarge image to make size 4:5'),
    (a45p, 'enlarge image to make size 4:5 (force portrait)'),
    (a45s, 'enlarge image to make size 4:5 (force landscape)'),
    (a4, 'enlarge image to make size A4 (1:sqrt(2))'),
    (a4p, 'enlarge image to make size A4 portrait (1:sqrt(2))'),
    (a4s, 'enlarge image to make size A4 landscape (sqrt(2):1)'),
    (ap2, 'enlarge image to make size a power of 2'),
    (e5, 'enlarge image with 5% border'),
    (e10, 'enlarge image with 10% border'),
    (e20, 'enlarge image with 20% border'),
    (e30, 'enlarge image with 30% border'),
    (sz, 'change canvas size to size specified with -w and -h, specify original image offset with -x and -y'),
    (sp2, 'change canvas size to the next power of 2, specify alignment with --halign and --valign'),
    (cjpg, 'convert to JPEG'),
    (cpng, 'convert to PNG'),
    (ctga, 'convert to TGA'),
    (rsz, 'resize image, specify width and/or height with -w and -h'),
    (r25, 'resize image to 25%'),
    (r50, 'resize image to 50%'),
    (r75, 'resize image to 75%'),
    (r200, 'resize image to 200%'),
    (h90, 'resize to 90% height'),
    (h95, 'resize to 95% height'),
    (w90, 'resize to 90% width'),
    (w95, 'resize to 95% width'),
    (mir, 'mirror image horizontally'),
    (flp, 'flip image vertically'),
    (b5, 'brighten image by 5%'),
    (b10, 'brighten image by 10%'),
    (b20, 'brighten image by 20%'),
    (b50, 'brighten image by 50%'),
    (b100, 'brighten image by 100%'),
    (d5, 'darken image by 5%'),
    (d10, 'darken image by 10%'),
    (d20, 'darken image by 20%'),
    (gb2, 'gaussian blur image 2 pixels radius'),
    (gb4, 'gaussian blur image 4 pixels radius'),
    (gb8, 'gaussian blur image 8 pixels radius'),
    (gb12, 'gaussian blur image 12 pixels radius'),
    (sh, 'sharpen image'),
    (kr, 'crop image, specify x1/y1/x2/y2 with -r'),
]

# dictionary for quick lookup of operation name to operation
opers_map = dict([(op[0].__name__, op[0]) for op in operlist])

# these process multiple input images
multi_opers = {t2, t3, t4, t6, t8, t9, t12, idx}


class Settings(object):
    def __init__(self, opts=None):
        if not opts:
            opts = {}
        self.frame = 0
        self.cutoff = 5
        self.output = None
        self.crop_rect = None
        self.bgcolor = '#ffffff'
        self.x = 0
        self.y = 0
        self.height = 0
        self.width = 0
        self.overwrite = False
        self.strip_icc_profile = False
        self.format = 'JPEG'
        self.halign = 0
        self.valign = 0
        self.save_opts = {'quality': 95}
        if opts:
            self.parse(opts)

    def parse(self, opts):
        for o, a in opts:
            if o == '-q':
                self.save_opts.quality = int(a)
                print('Setting quality to', self.save_opts.quality)
            elif o == '-b':
                self.bgcolor = None if a.lower() == 'none' else a
                print('Setting background color to', self.bgcolor)
            elif o == '-f':
                self.frame = int(a)
                print('Setting frame to', self.frame)
            elif o == '-c':
                self.cutoff = int(a)
                print('Setting cutoff to', self.cutoff)
            elif o == '-o':
                self.output = a
                print('Setting output filename to', self.output)
            elif o == '-r':
                m = re.match(r'(\d+)/(\d+)/(\d+)/(\d+)', a)
                if m:
                    self.crop_rect = [int(g) for g in m.groups()]
                    print('Setting crop rectangle to', a)
                else:
                    print('Invalid crop rectangle, expected x1/y1/x2/y2')
            elif o == '-x':
                self.x = int(a)
            elif o == '-y':
                self.y = int(a)
            elif o == '-h':
                self.height = int(a)
            elif o == '-w':
                self.width = int(a)
            elif o == '--overwrite':
                self.overwrite = True
            elif o == '--strip_icc_profile':
                self.strip_icc_profile = True
            elif o == '--halign':
                self.halign = int(a)
            elif o == '--valign':
                self.valign = int(a)

    def get_save_opts(self):
        save_opts = self.save_opts.copy()
        if self.format == 'TGA' and save_opts['quality'] < 100:
            save_opts['rle'] = True
        return save_opts


class FileName(object):
    c = re.compile(r'(.+)(\..+)')    # splits filename into name and extension

    def __init__(self, path):
        self.path = path
        m = self.c.match(path)
        self.base = m.group(1) if m else path
        self.ext = m.group(2) if m else ''

    def new_path(self, postfix):
        return self.base + postfix + self.ext

    def name(self):
        head, tail = ntpath.split(self.path)
        return tail or ntpath.basename(head)


def expand_paths(args):
    files = [glob.glob(arg) for arg in args]
    return sum(files, [])


def exec_multi_op(mop, files, settings):
    if settings.output:
        files = [f for f in files if f != settings.output]
    if mop == idx:
        # exclude already tiled images from the index
        cc = re.compile('_([xt][469]|idx)')
        files = [f for f in files if not cc.search(f)]
    ims = [Image.open(f) for f in files]
    im = mop(ims, settings)
    if im:
        fmt = settings.format
        output = settings.output
        if not output:
            filename = FileName(files[0])
            postfix = '_' + mop.__name__
            output = filename.new_path(postfix)
        save_opts = settings.get_save_opts()
        im.save(output, fmt, **save_opts)
    else:
        print('No output')


def exec_single_ops(sops, files, settings):
    for fpath in files:
        all_postfix = ''
        try:
            im = Image.open(fpath)
            # print 'Information:'
            # print im.info
            fmt = im.format
            new_im = None
            filename = FileName(fpath)
            fname = filename.name()
            for sop in sops:
                postfix = ''
                if not settings.overwrite:
                    postfix = '_' + sop.__name__
                    if postfix in fname:
                        continue    # apply operation only once
                im2 = sop(new_im or im, settings)
                if im2:
                    new_im = im2
                    if sop.__name__[0] == 'c':
                        fmt = settings.format or 'JPEG'
                        ext = '.' + ('jpg' if fmt == 'JPEG' else fmt.lower())
                        if filename.ext != ext:
                            filename.ext = ext
                            if settings.frame == 0:
                                postfix = ''
                            elif not settings.overwrite:
                                postfix += str(settings.frame)
                    all_postfix += postfix
            if new_im:
                if settings.output and len(files) == 1:
                    new_name = settings.output
                else:
                    new_name = filename.new_path(all_postfix)
                save_opts = settings.get_save_opts()
                new_im.save(new_name, fmt, **save_opts)
            else:
                print('No output for', fpath)
        except Exception as err:
            if isinstance(err, TypeError):
                import traceback
                traceback.print_exc()
            else:
                print(fpath, ': ', type(err), str(err))


def main(argv):
    def usage(show_opers=False):
        print('ImgOp 1.0 Copyright 2010-2022 Rene Smit')
        if show_opers:
            print('Operations:')
            for o, desc in operlist:
                print('  %-4s: %s' % (o.__name__, desc))
        else:
            print(usage_string)
        sys.exit(2)

    if len(argv) < 2:
        usage()
    elif len(argv) == 2 and argv[1].lower() == 'list':
        usage(True)

    opers = argv[1].split(',')
    if len(opers) == 1:
        if not opers[0] in opers_map:
            usage()
    else:
        for opname in opers:
            oper = opers_map.get(opname)
            if not oper or oper in multi_opers:
                usage()

    try:
        params = argv[2:]
        opts, args = getopt.getopt(params,
                                   'q:b:f:c:o:r:x:y:w:h:',
                                   ['overwrite', 'strip_icc_profile', 'halign=', 'valign='])
        if len(args) == 0:
            usage()

        settings = Settings(opts)
        files = expand_paths(args)
        ops = [opers_map[opname] for opname in opers]
        if len(ops) == 1 and ops[0] in multi_opers:
            # multi_op: (operation that needs multiple input files)
            exec_multi_op(ops[0], files, settings)
        else:
            # single_op: a single operation, or multiple operations applied to all input files
            exec_single_ops(ops, files, settings)
    except getopt.GetoptError as err:
        print(str(err))
        usage()


if __name__ == "__main__":
    main(sys.argv)
