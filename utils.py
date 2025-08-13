import tarfile

from os import path, remove
from urllib import request


def download_file(url, local_path="."):
  file_name = url.split("/")[-1]
  file_path = path.join(local_path, file_name)

  with request.urlopen(request.Request(url), timeout=30.0) as response:
    if response.status == 200:
      with open(file_path, "wb") as f:
        f.write(response.read())
  return file_path


def download_extract(url, target_path):
  tar_path = download_file(url)

  tar = tarfile.open(tar_path, "r:gz")
  tar.extractall(target_path, filter="data")
  tar.close()
  remove(tar_path)


def download_image(img_url, id):
  image_url = f"{img_url}/full/{id}.jpg"
  filename = f"{img_url}/{id}.jpg"
  request.urlretrieve(image_url, filename)


def constrain(v, vmin, vmax):
  return max(vmin, min(v, vmax))


def boxpct2pix(box, dim):
  (x0,y0,x1,y1), (w,h) = box, dim
  return (
    constrain(int(x0 * w), 0, w),
    constrain(int(y0 * h), 0, h),
    constrain(int(x1 * w), 0, w),
    constrain(int(y1 * h), 0, h),
  )


def centerpct2boxpix(cpct, len, dim):
  (cx, cy), (cw,ch), (w,h) = cpct, len, dim
  return (
    constrain(int(cx * w - cw / 2), 0, w),
    constrain(int(cy * h - ch / 2), 0, h),
    constrain(int(cx * w + cw / 2), 0, w),
    constrain(int(cy * h + ch / 2), 0, h),
  )
