import cv2
import numpy as np
import os
import Queue
import urllib2
import threading
import shutil

from multiprocessing import Pool
from PIL import Image
from StringIO import StringIO
from argparse import ArgumentParser

DEBUG_MODE = False
SAVE_IMAGE = True

# Usage:
# python 8bitter.py -p 1 -s 400 -o 1920,1080
# python 8bitter.py -s 600 -o 600,600 -n 100 -b COLOR_BG -e GLOW -r 1
# usage: 8bitter.py [-h] -s LOGOLEN -o OUTSZ [-p POKEINDEX] [-b BGTYPE]
#                  [-e EFFECT] [-n NUM_BLOCKS] [-f OFFSET] [-r BRIGHT]
#
#8bitter.py [Args] [Options] Detailed options -h or --help
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -s LOGOLEN, --8bitsize LOGOLEN
#                        Side length of the 8 bit img.
#  -o OUTSZ, --outsize OUTSZ
#                        Size of the output image. (Height,Width)
#  -p POKEINDEX, --pokeNum POKEINDEX
#                        Pokemon index number
#  -b BGTYPE, --bgtype BGTYPE
#                        Background style. NONE|COLOR_BG. NONE: No background.
#                        COLOR_BG: Colored background.
#  -e EFFECT, --effect EFFECT
#                        Effect. NONE|GLOW|OUTLINE. NONE: No effects.GLOW:
#                        Glowing outline. OUTLINE: Black outline.
#  -n NUM_BLOCKS, --numblocks NUM_BLOCKS
#                        Number of blocks for the height
#  -f OFFSET, --offset OFFSET
#                        8 bit image offset
#  -r BRIGHT, --bright BRIGHT
#                        Background color brightness level. 0-8

def poke_downloader(index):
    # download the image from the website
    img_url = 'http://pokemon.symphonic-net.com/'+index+'.gif'
    print index
    #print 'Downloading image from ' + img_url

    fd = urllib2.urlopen(img_url)
    im = Image.open(StringIO(fd.read()))
    im.save('tmp/'+index+'.png')

def bit_converter(f,args):
    # configure params
    print f
    BGTYPE = args.BGTYPE
    EFFECT = args.EFFECT
    bright = args.bright

    logoLen = args.logoLen
    logoSz = (logoLen,logoLen,3)

    outSz = args.outSz + ',3'
    outSz = tuple([int(a) for a in outSz.split(',')])

    num_blocks = args.num_blocks

    offsets = [int(a) for a in args.offset.split(',')]
    vOffset = offsets[0]
    hOffset = offsets[1]

    black = (0,0,0)

    img = cv2.imread(f)
    (hei,wid,alpha) = img.shape

    # figure out the padding
    oriColor = img[0,0,:]
    leftC = 0
    rightC = 0
    upC = 0
    downC = 0
    while np.all(np.abs(img[:,leftC,:]-oriColor)<10):
        leftC += 1
    while np.all(np.abs(img[:,wid-rightC-1,:]-oriColor)<10):
        rightC += 1
    while np.all(np.abs(img[upC,:,:]-oriColor)<10):
        upC += 1
    while np.all(np.abs(img[hei-downC-1,:,:]-oriColor)<10):
        downC += 1

    img = img[upC:hei-downC,leftC:wid-rightC]

    (hei,wid,alpha) = img.shape    

    padding = 100
    sideLen = np.floor(np.maximum(hei,wid))+padding
    tmp_img = np.ones((sideLen,sideLen,3),np.uint8)*oriColor
    x = np.floor((sideLen-wid)/2)
    y = np.floor((sideLen-hei)/2)
    tmp_img[y:y+hei,x:x+wid,:] = img
    img = tmp_img

    block_len = sideLen/num_blocks
    # end param configs

    # determine the background
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray,246,255,
                              cv2.THRESH_BINARY_INV)

    bg_color = cv2.mean(img,mask=mask)[0:3]
    bg_color /= np.max(bg_color)
    if np.min(cv2.mean(img,mask=mask)[0:3])<140:
        img = cv2.bitwise_and(img,img,mask=mask)

    # create the block image
    block_img = np.zeros((num_blocks,num_blocks,3),np.uint8)
    for vblock in range(num_blocks):
        h = np.floor(block_len*vblock)
        for wblock in range(num_blocks):
            w = np.floor(block_len*wblock)

            bg_count = np.sum(img[h:h+block_len,w:w+block_len,:]==(0,0,0))
            bg_count2 = np.sum(img[h:h+block_len,w:w+block_len,:]==oriColor)
            if bg_count > num_blocks**2*0.3 or bg_count2 > num_blocks**2*0.3:
                continue

            for c in range(3):
                block_img[vblock,wblock,c] \
                        = np.median(img[h:h+block_len,w:w+block_len,c])

    oriColor = np.mean(img[0,0,:].astype('int32'))
    block_img32 = block_img.astype('int32')

    # outline the 8-bit output
    s = ((-1,0),(1,0),(0,-1),(0,1))
    outerQ = Queue.Queue()
    outerQ.put((0,0))
    block_img[0,0,:] = black
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY).astype('int32')
    outer_img = np.zeros((num_blocks,num_blocks,3),np.uint8)
    outList = []
    inner_img = np.zeros((num_blocks,num_blocks,3),np.uint8)
    visitMat = np.zeros((num_blocks,num_blocks),np.uint8)
    visitMat[0,0] = 1
    while not outerQ.empty():
        (ver,hor) = outerQ.get()
        for i in range(4):
            v = ver+s[i][0]
            h = hor+s[i][1]
            if v<0 or h<0 or v>num_blocks-1 \
                    or h>num_blocks-1 or visitMat[v,h]:
                continue
            if np.abs(oriColor-gray[v,h])<10 and visitMat[v,h]==0:
                outerQ.put((v,h))
                visitMat[v,h] = 1
                block_img[v,h,:] = black
            else:
                outer_img[ver,hor,:] = black
                inner_img[v,h,:] = block_img[v,h,:]
                outList.append((v,h))

    fore_img_logo = np.zeros(logoSz,np.uint8)
    block_stretch = float(logoLen)/num_blocks
    for h in range(num_blocks):
        h2 = np.floor(h*block_stretch)
        h3 = np.ceil(h2+block_stretch)
        for w in range(num_blocks):
            w2 = np.floor(w*block_stretch)
            w3 = np.ceil(w2+block_stretch)
            if np.all(block_img[h,w,:]!=black):
                fore_img_logo[h2:h3,w2:w3,:] = block_img[h,w,:]

    # enlarge the fore img
    fore_img = np.zeros(outSz,np.uint8)
    fore_img[(outSz[0]-logoLen-vOffset):outSz[0]-vOffset, \
             (outSz[1]-logoLen-hOffset):outSz[1]-hOffset,:] \
                    = fore_img_logo

    # create the background img
    bg_mask = np.all((fore_img==(0,0,0)),2)
    bg_mask = (bg_mask*255).astype('uint8')

    if EFFECT == 'NONE':
        eff_img = fore_img

    elif EFFECT == 'OUTLINE':
        outline_img = np.zeros((num_blocks,num_blocks,3),np.uint8)
        outline_color = bg_color*255
        for pt in outList:
            v = pt[0]
            h = pt[1]
            block_img[v,h,:] = outline_color.astype('uint8')
        tmp_img = cv2.resize(block_img,(logoLen,logoLen),
                             interpolation=cv2.INTER_AREA)

        eff_img = np.zeros(outSz,np.uint8)
        eff_img[(outSz[0]-logoLen-vOffset):outSz[0]-vOffset, \
                (outSz[1]-logoLen-hOffset):outSz[1]-hOffset,:] \
                        = tmp_img

    elif EFFECT == 'GLOW':
        # create a glow effect
        glow_img = cv2.GaussianBlur(fore_img,(201,201),0)
        glow_img = cv2.bitwise_and(glow_img,glow_img,mask=bg_mask)

        # combine the images
        eff_img = fore_img+glow_img

    if BGTYPE == 'NONE':
        img = eff_img
    elif BGTYPE == 'COLOR_BG':
        back_img = (np.ones(outSz,np.uint8)*bg_color*32*bright).astype('uint8')
        back_img = cv2.bitwise_and(back_img,back_img,mask=bg_mask)

        img = np.clip(eff_img.astype('uint32')+back_img,0,255)
        img = img.astype('uint8')
        
    if DEBUG_MODE:
        cv2.imshow('frame',img)
        cv2.waitKey()
    if SAVE_IMAGE:
        try:
            os.mkdir('output')
        except:
            pass
        cv2.imwrite('output/'+f.split('/')[-1],img)

if __name__ == '__main__':
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description=desc)

    parser.add_argument(
        '-s', '--8bitsize',
        type = int,
        dest = 'logoLen',
        required = True,
        help = 'Side length of the 8 bit img.'
    )
    parser.add_argument(
        '-o', '--outsize',
        type = str,
        dest = 'outSz',
        required = True,
        help = 'Size of the output image. (Height,Width)'
    )

    parser.add_argument(
        '-p', '--pokeNum',
        type = str,
        dest = 'pokeIndex',
        default = None,
        help = 'Pokemon index number'
    )
    parser.add_argument(
        '-b', '--bgtype',
        type = str,
        dest = 'BGTYPE',
        default = 'COLOR_BG',
        help = 'Background style. NONE|COLOR_BG. ' \
               + 'NONE: No background. COLOR_BG: Colored background.' 
    )
    parser.add_argument(
        '-e', '--effect',
        type = str,
        dest = 'EFFECT',
        default = 'GLOW',
        help = 'Effect. NONE|GLOW|OUTLINE. ' \
               + 'NONE: No effects.' \
               + 'GLOW: Glowing outline. OUTLINE: Black outline.'
    )
    parser.add_argument(
        '-n', '--numblocks',
        type = int,
        dest = 'num_blocks',
        default = 40,
        help = 'Number of blocks for the height'
    )
    parser.add_argument(
        '-f', '--offset',
        type = str,
        dest = 'offset',
        default = '0,0',
        help = '8 bit image offset'
    )
    parser.add_argument(
        '-r', '--bright',
        type = int,
        dest = 'bright',
        default = '2',
        help = 'Background color brightness level. 0-8'
    )

    args = parser.parse_args()

    if args.pokeIndex:
        pokeIndex = args.pokeIndex.split(',')
        pokeIndex = ['0'*(3-len(elem))+elem for elem in pokeIndex]

    # first download the images
    try:
        os.mkdir('tmp')
    except:
        pass

    pool = Pool()
    if args.pokeIndex:
        pool.map(poke_downloader,pokeIndex)
    
    # then synthesize 8-bit images
    for root, dirs, files in os.walk('tmp'):
        for f in files:
            pool.apply_async(bit_converter,args=('tmp/'+f,args,))
    for root, dirs, files in os.walk('imgs'):
        for f in files:
            pool.apply_async(bit_converter,args=('imgs/'+f,args,))
    pool.close()
    pool.join()
    
    #bit_converter('imgs/12.jpg',args)

    # delete the temp directory
    #shutil.rmtree(u'tmp')