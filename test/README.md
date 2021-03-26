
# test

## Code naming convention

- H: Height, x
- W: Width, y
- D: Depth, z

and we always uses x, y, z  in (x,y,z) order,


coordiante are in z,y,x format

we treate z as x, y as y, as x as z. when constructing our SparseTensor.
## weight channels covention

when comparing spconv and sphconv, we need permutation

we use, [H, W, D, outChannel, inChannel],
in which H, W, D correspond to x, y, z coorindates

in spconv, it uses [H, W, D, inChannel, outChannel],



But