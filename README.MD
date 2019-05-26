nyan256
====
Convert your image into VGA256 color plate and generate binarary file which can directly include in yout assembly project.

## Feature
Original image:  
![nyannnnn](https://i.imgur.com/nvvUT22.png)  
Converted image:  
![nyyyyyan](https://github.com/ifTNT/nyan256/raw/master/output.png)  
Disply in dosbox:  (use [MODE13H](https://github.com/ifTNT/MODE13H))  
![anny](https://github.com/ifTNT/nyan256/raw/master/dosbox.png) 

**With generated pixel-by-pixel binary file**

## Requirement
- Python^=3.7.3
- opencv-python
- numpy
  
or execute `pip install -r requirement.txt` in root of this project.

## Author / License
ifTNT @2019 / MIT

## Benchmark
RGB_24bits_palette_color_test_chart (from [Wikipedia](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#/media/File:RGB_24bits_palette_color_test_chart.png)):  
![24bit](https://upload.wikimedia.org/wikipedia/commons/b/b8/RGB_24bits_palette_color_test_chart.png)

Converted image:  
![24bit_256](https://github.com/ifTNT/nyan256/raw/master/RGB_24bits_256.png)

## Priciple

A VGA256 color plate may looks like  
![vga256](https://i.imgur.com/U0ux7Xt.png)
- 0x00 - 0x0F: Traditional 16 colors
- 0x10 - 0x1F: Grayscale
- 0x20 - 0xF7: Real 216 color
- 0xF8 - 0xFF: Unused

If rearranging segment 0x20-0xF7 with each row containing 24-colors. We will get the following color plate:  
![r256](https://i.imgur.com/3Ey4DSz.png)
  
I guessed it's a HSV color space. Luckly, I'm approximately right.  
So the next step is restrict normal HSV color space to this 24 * 3 * 3 HSV color space.  
And you'll get a nicely converted picture in VGA256-color mode.
