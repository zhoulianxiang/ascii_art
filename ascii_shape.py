#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


logger = logging.getLogger(__name__)


class AsciiTxt(object):
    '''Adjust: search 'Adjust.X' for Adjust point
    '''
    
    def __init__(self,
        txt='.oO',
        dst_height=24,
        width_fix=0,
        font_path='font/msyh.ttf',
        blank=' ',
        block='#',
        fast=False,
        char_set=None):
        '''
        Argvs:
            txt - 
            dst_height - output lines
            width_fix - dst_height is auto calculated, give a the coefficient to fix it
                - 0 or None, keep original scale from font
                - n > 0, dst_width = dst_width * n
                - n < 0, dst_width = dst_height * n * len(txt) * -1
            font_path - *.ttf file path
            blank - fill the 'white' block
            block - file the 'black' block
            fast - use faster 'Image similarity algorithms'
            char_set - use given ascii character set
        '''
        self.txt = txt
        self.dst_h = dst_height
        self.w_fix = width_fix
        self.font_path = font_path
        self.blank = blank
        self.block = block
        self.is_fast = fast
        self.char_set = char_set
        
        self.algo = self._cal_mse if self.is_fast else self._cal_ssim

        #Adjust point
        self.shape_font_size = 24
        self.shape_ksize = 9
        logger.debug(f'Shape font size: {self.shape_font_size}')
        logger.debug(f'Shape GaussionBlur ksize: {self.shape_ksize}')

        #ASCII Visible Characters, from 0x20~0x7F, exclude (0x20, 0x7f)
        #0x20 - Space,  we set self.blank specially
        #0x7f - DEL(Delete)
        if not self.char_set: self.char_set = [chr(_) for _ in range(33,127)]

        self.shapes = self._init_shapes()


    def _init_shape(self, c, font):
        #new img - draw text - resize - blur - ???
        img = Image.new('RGB', (self.sw, self.sh), color=(255,255,255))

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), c, fill=(0,0,0), font=font)

        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        #Adjust.2 shape_ksize
        blur_img = cv2.GaussianBlur(cv2_img, (self.shape_ksize, self.shape_ksize), 0)

        #Adjust.3 binarization it?
        #mask = ~(blur_img == [255,255,255]).all(axis=-1)
        #blur_img[mask] = [0, 0, 0]

        return blur_img


    def _init_shapes(self):
        #Adjust.1 shape_font_size
        font = ImageFont.truetype('font/msyh.ttf', self.shape_font_size)

        char_widths = set([font.getlength(_) for _ in self.char_set])
        if len(char_widths) != 1:
            logger.warn(f'Font of shape[{font.path}] is not monospace')
            return None

        self.sw = int(list(char_widths)[0]) 
        self.sh = sum(font.getmetrics())
        logger.debug(f'Shape img size: {self.sw}x{self.sh}')

        return {_: self._init_shape(_, font) for _ in self.char_set}

        
    def _gen_img(self):
        font = ImageFont.truetype(self.font_path, 512)
        logger.debug(f'text font metrice: {font.getmetrics()}')

        logger.debug(f'text bbox: {font.getbbox(self.txt)}')
        x, y, w, h = font.getbbox(self.txt)
        img_w = int(font.getlength(self.txt))
        img_h = h - y
        logger.debug(f'Text img size: {img_w}x{img_h}')
        if img_h == 0:
            logger.fatal('Text img_h == 0')
            return None

        img = Image.new('RGB', (img_w, img_h), color=(255,255,255))
        draw = ImageDraw.Draw(img)
        draw.text((0,-y), self.txt, fill=(0,0,0), font=font)

        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) 
        logger.debug(f'cv2 img shape: {cv2_img.shape}')

        #self.dst_h, in __init__()
        self.dst_w = img_w*self.dst_h//img_h
        if not self.w_fix:
            pass
        elif self.w_fix > 0:
            self.dst_w = int(self.dst_w*self.w_fix)
        elif self.w_fix < 0:
            self.dst_w = int(-1*self.dst_h*self.w_fix*len(self.txt))
        logger.debug(f'dst size: {self.dst_w}x{self.dst_h}')
        resize_img = cv2.resize(cv2_img, (self.dst_w*self.sw, self.dst_h*self.sh))
        logger.debug(f'resized img shape: {resize_img.shape}')

        #Adjust.4 blur it? ksize?
        #blur_img = cv2.GaussianBlur(resize_img, (self.ksize, self.ksize), 0)

        return resize_img


    def _cal_ssim(self, img, shape):
        #Adjust.5 win_size
        #Adjust.6 gaussian_weights
        return structural_similarity(img, shape, win_size=7, gaussian_weights=False)


    def _cal_mse(self, img, shape):
        return -1 * mean_squared_error(img, shape)


    def _cal_psnr(self, img, shape):
        '''for test
        '''
        return peak_signal_noise_ratio(img, shape)


    def _find_shape(self, x, y):
        img = self.img[y*self.sh:(y+1)*self.sh, x*self.sw:(x+1)*self.sw]

        # black or white? for fastly
        means = np.mean(img)
        if means > 250:
            return self.blank
        elif means < 5:
            return self.block

        scores = [(self.algo(img, self.shapes[_]), _) for _ in self.shapes]
        scores.sort()

        return scores[-1][1]


    def _match_shapes(self):
        if not self.shapes or self.img is None: return ''

        img_h, img_w = self.img.shape[:2]
        dst_h, dst_w = img_h//self.sh, img_w//self.sw
        img_txt = []
        for h in range(dst_h):
            img_txt.append(''.join([self._find_shape(w, h) for w in range(dst_w)]))

        return img_txt


    def __call__(self):
        if not self.shapes: return ''
        self.img = self._gen_img()
        return '\n'.join(self._match_shapes())


    def train(self):
        for c, sh in self.shapes.items():
            sim_scores = [(self.algo(sh, self.shapes[_]), _) for _ in self.shapes]
            sim_scores.sort(reverse=True)
            print(f'{c}:', ''.join([_[1] for _ in sim_scores]))
        return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('txt', nargs='?', type=str, default='.oO', help='input text, not too long')
    parser.add_argument('-H', '--dst_height', metavar='<h>', type=int, default=24, help='output height(lines)')
    parser.add_argument('-W', '--width_fix', metavar='<n>', type=float, default=0, help='0 or None: do nothing; > 0: relate to width; < 0: relate to height')
    parser.add_argument('-F', '--font_path', metavar='<path>', type=str, default='font/msyh.ttf', help='path of *.ttf. To find it from fonts folder of os system')
    parser.add_argument('--block', metavar='<c>', type=str, default='.', help='Character for black block')
    parser.add_argument('--blank', metavar='<c>', type=str, default=' ', help='Character for white block')
    parser.add_argument('-f', '--fast', action='store_true', default=False, help='use faster shape match algorithm')
    parser.add_argument('--char_set', metavar='<str>', type=str, default=None, help='Character settings, default 0x33-0x7e')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='show debug message')
    args = parser.parse_args()

    fonts = [
        'font/msyh.ttf',
        'font/lqhkxsjft.ttf',
        'font/fzybxsft.ttf'
    ]
    if args.font_path in map(str, range(len(fonts))): args.font_path = fonts[int(args.font_path)]
    if not os.path.exists(args.font_path): sys.exit(f'Font file({args.font_path}) does not exists.')

    print(args)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    at_args = vars(args)
    at_args.pop('debug', None)
    at = AsciiTxt(**at_args)

    '''test code
    for c, s in at.shapes.items():
        cv2.imwrite(f'shape_{c}.jpg', s)
    exit(0)
    '''

    '''test code
    at.train()
    exit(0)
    '''

    print(at())

