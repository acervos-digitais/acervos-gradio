import json

import gradio as gr
import numpy as np

from os import makedirs, path
from PIL import Image as PImage

from utils import download_extract, download_file, get_crop_filenames
from utils import boxpct2pix, centerpct2boxpix


OBJS_URLS = "https://raw.githubusercontent.com/acervos-digitais/herbario-data/main/json/20250705_processed.json"
IMG_URL = "https://digitais.acervos.at.eu.org/imgs/herbario/arts"
IMG_CROPS_DIR = "./imgs/crops"
IMG_FULL_DIR = "./imgs/full"
GRID_MIN_CROP_HEIGHT = 64
XY_OUT_DIM = (1024, 1024)
XY_CROP_MAX = 0.45
MAX_PIXELS = 2**25
PImage.MAX_IMAGE_PIXELS = 2 * MAX_PIXELS


def get_min_height_and_size(idBoxes_all, min_min_height=GRID_MIN_CROP_HEIGHT):
  heights = []
  sizes = {}

  for idBoxes in idBoxes_all:
    id = idBoxes["id"]

    img = PImage.open(path.join(IMG_FULL_DIR, f"{id}.jpg"))
    iw,ih = img.size
    sizes[id] = (iw,ih)

    for (x0,y0,x1,y1) in idBoxes["boxes"]:
      crop_h = ih * (y1 - y0)
      heights.append(max(crop_h, min_min_height))

  heights_sorted = list(set(sorted(heights)))

  return heights_sorted[0], sizes


def get_mosaic_size(idBoxes_all, height_min, sizes):
  width_sum = 0

  for idBoxes in idBoxes_all:
    id = idBoxes["id"]

    iw,ih = sizes[id]

    for (x0,y0,x1,y1) in idBoxes["boxes"]:
      crop_w = iw * (x1 - x0)
      crop_h = ih * (y1 - y0)
      width_sum += (height_min / crop_h) * crop_w

  total_area = width_sum * height_min
  scale = 1.0
  mos_w = total_area ** 0.5
  mos_h = 1.2 * mos_w

  if total_area > MAX_PIXELS:
    scale = (MAX_PIXELS / total_area) ** 0.5
    mos_w *= scale
    mos_h *= scale

  return int(mos_w), int(mos_h), scale


def get_crop_img(boxKey):
  if boxKey in box2fname:
    fname = box2fname[boxKey]
    cimg = PImage.open(path.join(IMG_CROPS_DIR, fname))
    crop_w, crop_h = cimg.size
  else:
    img = PImage.open(path.join(IMG_FULL_DIR, f"{id}.jpg")).convert("RGB")
    iw,ih = img.size
    src_x0, src_y0, src_x1, src_y1 = boxpct2pix((x0,y0,x1,y1), (iw,ih))
    cimg = img.crop((src_x0, src_y0, src_x1, src_y1))
    crop_w, crop_h = cimg.size
  return cimg, crop_w, crop_h


def get_objetcs_grid_mosaic(idBoxes_in):
  idBoxes_all = [x for x in idBoxes_in if len(x["boxes"]) > 0]

  height_min, sizes = get_min_height_and_size(idBoxes_all)
  mos_w, mos_h, limit_scale = get_mosaic_size(idBoxes_all, height_min, sizes)

  mos_img = PImage.fromarray(np.zeros((mos_h, mos_w))).convert("RGB")
  cur_x, cur_y = 0, 0

  for idBoxes in idBoxes_all:
    id = idBoxes["id"]

    for (x0,y0,x1,y1) in idBoxes["boxes"]:
      boxKey = (id,x0,y0,x1,y1)
      cimg, crop_w, crop_h = get_crop_img(boxKey)

      scale_factor = limit_scale * (height_min / crop_h)
      crop_w, crop_h = int(scale_factor * crop_w), int(scale_factor * crop_h)

      crop_img = cimg.resize((crop_w, crop_h))

      if cur_y >= mos_h:
        print("break")
        break

      mos_img.paste(crop_img, (cur_x, cur_y))
      cur_x += crop_w

      if cur_x > mos_w:
        overflow_x = cur_x - mos_w
        crop_img = crop_img.crop((crop_w - overflow_x, 0, crop_w, crop_h))
        cur_x = 0
        cur_y += crop_h
        mos_img.paste(crop_img, (cur_x, cur_y))
        cur_x += overflow_x

  if cur_x < mos_w and cur_y < mos_h:
    empty_w = mos_w - cur_x
    row = mos_img.crop((0, 0, empty_w, height_min))
    mos_img.paste(row, (cur_x, cur_y))

  mos_img = mos_img.crop((0, 0, mos_w, cur_y + crop_h))
  mos_img.thumbnail((2048, 2048))
  return mos_img


def get_xy_mosaic(idBoxes_in):
  idBoxes_all = [x for x in idBoxes_in if len(x["boxes"]) > 0]

  pix_cnts = np.zeros(XY_OUT_DIM)
  pix_vals = np.zeros((*XY_OUT_DIM, 3))

  for idBoxes in idBoxes_all:
    id = idBoxes["id"]

    img = PImage.open(path.join(IMG_FULL_DIR, f"{id}.jpg")).convert("RGB")
    iw,ih = img.size
    w_scale, h_scale = XY_OUT_DIM[0] / iw, XY_OUT_DIM[1] / ih
    crop_scale = min(w_scale, h_scale)
    siw, sih = iw * crop_scale, ih * crop_scale

    for (x0,y0,x1,y1) in idBoxes["boxes"]:
      crop_w = (x1 - x0)
      crop_h = (y1 - y0)

      if crop_w > XY_CROP_MAX or crop_h > XY_CROP_MAX:
        continue

      center_x = (x0 + x1) / 2
      center_y = (y0 + y1) / 2

      dst_w = int(crop_w * siw)
      dst_h = int(crop_h * sih)

      src_x0, src_y0, src_x1, src_y1 = boxpct2pix((x0,y0,x1,y1), (iw,ih))
      dst_x0, dst_y0, dst_x1, dst_y1 = centerpct2boxpix((center_x, center_y), (dst_w, dst_h), XY_OUT_DIM)

      dst_w = dst_x1 - dst_x0
      dst_h = dst_y1 - dst_y0

      crop_vals = np.array(img.crop((src_x0, src_y0, src_x1, src_y1)).resize((dst_w, dst_h)))
      pix_vals[dst_y0:dst_y1, dst_x0:dst_x1] += crop_vals
      pix_cnts[dst_y0:dst_y1, dst_x0:dst_x1] += 1

  pix_cnts_div = np.expand_dims(pix_cnts.copy(), axis=-1)
  pix_cnts_div[pix_cnts == 0] = 1

  pix_avg = pix_vals / pix_cnts_div
  return PImage.fromarray(pix_avg.astype(np.uint8))


### prep files and dirs
makedirs(IMG_CROPS_DIR, exist_ok=True)
makedirs(IMG_FULL_DIR, exist_ok=True)

download_extract(f"{IMG_URL}/crops.tgz", "./imgs")
download_extract(f"{IMG_URL}/full.tgz", "./imgs")

box2fname = get_crop_filenames(OBJS_URLS)


### start Gradio
with gr.Blocks() as demo:
  gr.Interface(
    title="objects",
    api_name="objects",
    fn=get_objetcs_grid_mosaic,
    inputs="json",
    outputs="image",
    flagging_mode="never",
  )

  gr.Interface(
    title="xy",
    api_name="xy",
    fn=get_xy_mosaic,
    inputs="json",
    outputs="image",
    flagging_mode="never",
  )

if __name__ == "__main__":
  demo.launch()
