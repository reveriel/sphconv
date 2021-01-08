// storage format for feature Map
// featureMap logically is a 5-order tensor of
//     B H W D C
// in which B is batchsize
// H W D are 3D spatial dimensions
// C is the channel dimension

// D is a sparse mode, store it in a compressed way
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

/**
 * @brief
 *
 * @tparam T
 * @tparam Index
 * @tparam B_
 * @tparam H_
 * @tparam W_
 * @tparam D_
 * @tparam C_
 */
template <typename T,
          typename Index,
          int B_,
          int H_,
          int W_,
          int D_,
          int C_
          >
struct Feature
{
    // NNZ denote the number of non-zero elements

    static unsigned int const B = B_;
    static unsigned int const H = H_;
    static unsigned int const W = W_;
    static unsigned int const D = D_;
    static unsigned int const C = C_;

    using ChannelVec = T[C];
    // all nonzeros are in the 'val'
    // size: NNZ * C
    // the 'val' stores all nonzeros
    // its a pointer points to an array of
    ChannelVec *val;

    // the 'z_ind' stores the z indexes of the elements in 'val'
    // if val[k] = A[b,x,y,z,c], then z_ind[k]  = z
    // size: NNZ
    Index *z_ind; // all z indeices

    // the 'z_ptr' stores the locations in 'val' that start a 'C' vector.
    // if val[k] = A[b,x,y,z,c], then  z_ptr[b,x,y] <= k < z_ptr[b,x,y + 1]
    // size: B * H * W
    Index z_ptr[B][H][W];
    // the 'nnz' also act as a guard, z_ptr[B-1][H-1][W] read this
    // WARN: this field must be next to z_ptr
    Index nnz;

    // invariant: capacity * C >= nnz
    uint32_t capacity;
    static const uint32_t MIN_capacity = 8;

private:
    /**
     * @brief expand the capacity
     */
    void expand() {
        if (capacity == 0) {
            // first time allocate
            capacity = MIN_capacity;
            val = (T (*) [C]) malloc(sizeof(T) * capacity * C);
            z_ind = (Index *)malloc(sizeof(Index) * capacity);
            return;
        }
        capacity *= 2;
        val = (T (*) [C]) realloc((void*)val, sizeof(T) * capacity * C);
        z_ind = (Index *)realloc((void*)z_ind, sizeof(Index) * capacity);
        return;
    }

    bool full() { return capacity * C >= nnz; }

public:
    Feature() : nnz(0), capacity(0), val(nullptr), z_ind(nullptr) {
        memset(z_ptr, 0, sizeof(z_ptr));
        assert(&nnz == &z_ptr[B-1][H-1][W]);
    }

    ~Feature() {
        if (val)
            free(val);
        if (z_ind)
            free(z_ind);
    }

    /**
     * @brief  get element, assume b,x,y,z,c is a valid index
     */
    T get(Index b, Index x, Index y, Index z, Index c) const {
        Index z_start = z_ptr[b][x][y];
        Index z_end = z_ptr[b][x][y+1];
        // find equal z
        Index i = z_start;
        for (i = z_start; i < z_end; i++)
            if (z_ind[i] == z)
                break;
        return val[i][c];
    }

    T &get(Index b, Index x, Index y, Index z, Index c) {
        Index z_start = z_ptr[b][x][y];
        Index z_end = z_ptr[b][x][y+1];
        // find equal z
        Index i = z_start;
        for (i = z_start; i < z_end; i++)
            if (z_ind[i] == z)
                break;
        return val[i][c];
    }


    bool has_idx(Index b, Index x, Index y, Index z, Index c) {
        if (b >= B || x >= H || y >= W || z >= D || c >= C) return false;
        if (nnz == 0) return false;
        Index z_start = z_ptr[b][x][y];
        Index z_end = z_ptr[b][x][y+1];
        for (Index i = z_start; i < z_end; i++)
            if (z_ind[i] == z)
                return true;
        return false;
    }

    void set(Index b, Index x, Index y, Index z, Index c, T value) {
        if (has_idx(b,x,y,z,c)) {
            this->get(b,x,y,z,c) = value;
            return;
        }
        if (full()) {
            expand();
        }
        Index z_start = z_ptr[b][x][y];
        Index z_end = z_ptr[b][x][y+1];
        // TODO: binary search
        Index i = z_start;
        for(; i < z_end; i++)
            if (z_ind[i] >= z)
                break;
        // shift
        for (Index j = nnz; j > i; j--) {
            z_ind[j] = z_ind[j-1];
        }
        z_ind[i] = z;
        val[i][c] = value;
        for (Index *p = &z_ptr[b][x][y] + 1; p != &nnz; p++)
            *p += 1;
        nnz += 1;
        return;
    }

    void print() {
        for (int b = 0; b < B; b++) {
            for (int x = 0; x < H; x++) {
                for (int y = 0; y < W; y++) {
                    for (Index start = z_ptr[b][x][y],
                               end = z_ptr[b][x][y+1];
                         start < end; start++)
                    {
                        for (int c = 0; c < C; c++) {
                            printf("[%d,%d,%d,%d,%d] -> %.2f\n",
                            b, x, y, z_ind[start], c, val[start][c]);
                        }
                    }
                }
            }
        }
    }

};




/**
 * @brief
 *
 * @param a
 */
void init(int a)
{
    // 1 x 3 x 3 x 3 x 16
}
