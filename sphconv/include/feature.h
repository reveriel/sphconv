// storage format for feature Map
// featureMap logically is a 5-order tensor of
//     B H W D C
// in which B is batchsize
// H W D are 3D spatial dimensions
// C is the channel dimension

// D is a sparse mode, store it in a compressed way

template<typename T, // data type
 typename Index //  index type
 >
struct Feature {
    private:
    T *data;
    Index **ind;  // all z indeices
    int **ptr;

};

