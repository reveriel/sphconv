
# test

## Code naming convention

- H: Height, x
- W: Width, y
- D: Depth, z

and we always try to uses x, y, z  in (x,y,z) order, or  HWD order

spconv use zyx format for unknown reason.
coordiante are in z,y,x format,  since they are from spconv code.

## weight channels covention

in spconv, it uses [D, W, H, inChannel, outChannel],

same here


## BUT!

we treat zyx as xyz in c++/CUDA code !!!

记住, Python 这边 DW是稠密维度
c++ 那边 HW 是稠密维度