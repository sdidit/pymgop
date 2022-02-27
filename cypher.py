import random
import string

from PIL import Image, ImageDraw, ImageFont


def create_cyphertext(word, population, count):
    word_indices = list(zip(word, sorted(random.sample(range(count), len(word)))))
    # print(word_indices)
    cyphertext = random.choices(population, k=count)
    for c, offset in word_indices:
        cyphertext[offset] = c
    return cyphertext, word_indices


def main():
    # settings
    size = 10
    font_name = 'arial.ttf'
    font_size = 35
    population = string.digits  # string.ascii_uppercase
    word = '1234'
    filename = 'cypher'
    bgcolor = 'white'
    key_bgcolor = '#ccc'
    fgcolor = 'black'
    spacing = 16
    square = True

    # generate cyphertext
    cyphertext, solution = create_cyphertext(word, population, size * size)

    # determine sizes
    font = ImageFont.truetype(font_name, size=font_size)
    draw = ImageDraw.Draw(Image.new('RGB', (0, 0)))
    textsizes = [draw.textsize(c, font=font, spacing=0) for c in population]
    w = max(ts[0] for ts in textsizes) + spacing
    h = max(ts[1] for ts in textsizes) + spacing
    if square:
        w = h = max(w, h)
    edge_spacing = spacing * 2
    hspacing = spacing // 2
    width = size * w + edge_spacing
    height = size * h + edge_spacing

    # create cypher image
    cypher_im = Image.new('RGB', (width, height), color=bgcolor)
    draw = ImageDraw.Draw(cypher_im)
    for row in range(size):
        for col in range(size):
            i = row * size + col
            c = cyphertext[i]
            cw, ch = draw.textsize(c, font=font, spacing=0)
            x = edge_spacing // 2 + col * w + (w - cw) // 2
            y = edge_spacing // 3 + row * h + (h - ch) // 2
            draw.text((x, y), c, fill=fgcolor, font=font)
    cypher_im.save(f'{filename}.png', 'PNG')

    # create key image
    key_im = Image.new('RGBA', (width, height), color=key_bgcolor)
    hole_im = Image.new('RGBA', (w - hspacing, h - hspacing))
    for _, offset in solution:
        row = offset // size
        col = offset % size
        x = edge_spacing // 2 + col * w + hspacing // 2
        y = edge_spacing // 3 + row * h + hspacing // 2
        key_im.paste(hole_im, (x, y))
    key_im.save(f'{filename}_key.png', 'PNG')

    # show solution
    cypher_im.paste(key_im, mask=key_im)
    cypher_im.save(f'{filename}_solution.png', 'PNG')
    # cypher_im.show('Solution')


if __name__ == "__main__":
    main()
