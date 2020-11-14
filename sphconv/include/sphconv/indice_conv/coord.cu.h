#pragma once

#include "sphconv/sphconv.h"
#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace sphconv
{

using cutlass::Coord;

/// Helper to make a 4-element coordinate
CUTLASS_HOST_DEVICE
Coord<5> make_Coord(int _0, int _1, int _2, int _3, int _4) {
  int values[5] = {_0, _1, _2, _3, _4};
  return Coord<5>(values);
}

struct TensorNTHWCCoord : public cutlass::Coord<5, int> {

  using Base = Coord<5, int>;

  using Index = typename Base::Index;
  using LongIndex = typename Base::LongIndex;

  // dimensions
  static int const kN = 4;
  static int const kT = 3;
  static int const kH = 2;
  static int const kW = 1;
  static int const kC = 0;

  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord() { }

  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord(Base const &coord): Base(coord) { }

  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord(Index n, Index t, Index h, Index w, Index c) :
    Base(make_Coord(n, t, h, w, c)) { }

  // view w c as a single dimension
  TensorNTHWCCoord(Index h, Index wc) :
    Base(make_Coord(0, 0, h, 0, wc)) { }


  CUTLASS_HOST_DEVICE
  Index const & n() const { return this->at(kN); }

  CUTLASS_HOST_DEVICE
  Index  & n()  { return this->at(kN); }

  CUTLASS_HOST_DEVICE
  Index const & t() const { return this->at(kT); }

  CUTLASS_HOST_DEVICE
  Index  & t()  { return this->at(kT); }

  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index  & h()  { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  CUTLASS_HOST_DEVICE
  Index  & w()  { return this->at(kW); }

  CUTLASS_HOST_DEVICE
  Index const & c() const { return this->at(kC); }

  CUTLASS_HOST_DEVICE
  Index  & c()  { return this->at(kC); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord operator+(Base const& b) const {
    return TensorNTHWCCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord operator-(Base const& b) const {
    return TensorNTHWCCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord operator*(Base const& b) const {
    return TensorNTHWCCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord operator/(Base const& b) const {
    return TensorNTHWCCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  TensorNTHWCCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }

};

} // namespace sphconv
