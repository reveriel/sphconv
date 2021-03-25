
# test

## channel naming convention

- H: Height, x
- W: Width, y
- D: Depth, z

and we always uses x, y, z  in (x,y,z) order,
while in spconv it uses (z,y,x) format, for unknown reason.

## weight channels covention

when comparing spconv and sphconv, Note

we use, [H, W, D, outChannel, inChannel],
in which H, W, D correspond to x, y, z coorindates

in spconv, it uses [H, W, D, inChannel, outChannel],

spconv,   always

