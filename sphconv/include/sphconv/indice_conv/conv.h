#pragma once

#include "sphconv/sphconv.h"
#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace sphconv
{



/// TileCoord is a structure derived from Coord<2> that specifies a location within the
/// coordinate space of a indice_conv_tile problem.
// the top left corrnner
struct TileCoord : public cutlass::Coord<2, int> {
  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=2
  typedef Coord<2, Index> Base;

  /// GEMM H dimension - rows of the Input I matrix
  static int const kH = 0;

  /// GEMM W dimension - columns of the Input I matrix
  static int const kW = 1;

  CUTLASS_HOST_DEVICE
  TileCoord() { }

  CUTLASS_HOST_DEVICE
  TileCoord(Coord<2, Index> const &coord): Base(cutlass::make_Coord(coord[0], coord[1])) { }

  CUTLASS_HOST_DEVICE
  TileCoord(Index h, Index w): Base(cutlass::make_Coord(h, w )) { }

  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  TileCoord operator+(Base const& b) const {
    return TileCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  TileCoord operator-(Base const& b) const {
    return TileCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  TileCoord operator*(Base const& b) const {
    return TileCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  TileCoord operator/(Base const& b) const {
    return TileCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  TileCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  TileCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  TileCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  TileCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }

};


/// BatchedTileCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a batched indice conv problem.
struct BatchedTileCoord : public cutlass::Coord<3, int> {

  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=3
  typedef Coord<3, Index> Base;

  /// GEMM H dimension - rows of the input I matrix
  static int const kH = 0;

  /// GEMM W dimension - columns of the input I matrix
  static int const kW = 1;

  /// GEMM batch dimension - inner dimension of the GEMM problem
  static int const kBatch = 2;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  BatchedTileCoord() { }

  /// Constructs from Coord<4>
  CUTLASS_HOST_DEVICE
  BatchedTileCoord(Base const &coord): Base(coord) { }

  /// Helper to construct from a H, W, and batch variables
  CUTLASS_HOST_DEVICE
  BatchedTileCoord(Index h, Index w, Index b): Base(cutlass::make_Coord(h, w, b)) { }

  /// Returns the H coordinate
  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  /// Returns reference to the H coordinate
  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  /// Returns the W coordinate
  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  /// Returns reference to the W coordinate
  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  /// Returns the batch coordinate
  CUTLASS_HOST_DEVICE
  Index const & batch() const { return this->at(kBatch); }

  /// Returns reference to the batch coordinate
  CUTLASS_HOST_DEVICE
  Index & batch() { return this->at(kBatch); }

  /// Obtains a TileCoord from BatchedTileCoord
  CUTLASS_HOST_DEVICE
  TileCoord hw() const {
    return TileCoord(h(), w());
  }

  /// Obtains a Coord<4> from BatchedTileCoord
  CUTLASS_HOST_DEVICE
  Coord<3> hwb() const {
    return cutlass::make_Coord(h(), w(), batch());
  }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator+(Base const& b) const {
    return BatchedTileCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator-(Base const& b) const {
    return BatchedTileCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator*(Base const& b) const {
    return BatchedTileCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator/(Base const& b) const {
    return BatchedTileCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};


/// shape of the tile
template<
  int H = 4,
  int W = 4
>
struct TileShape {
  static int const kH = H;
  static int const kW = H;
  CUTLASS_HOST_DEVICE
  static Coord<2> toCoord() {
    return make_Coord(kH, kW);
  }
};



} // namespace sphconv