// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — ap int tests
//
// NON-SYNTHESIS lint / co-simulation shim for <ap_int.h>. This is NOT the AMD
// Xilinx header: it backs ap_int<W> / ap_uint<W> with two's-complement int64_t
// so the generated AXI4-Stream bundle compiles and runs under a host compiler
// for bit-true software co-simulation. Vivado/Vitis HLS supplies the authentic
// header at synthesis time.
#ifndef SCPN_HLS_SHIM_AP_INT_H
#define SCPN_HLS_SHIM_AP_INT_H

#include <cstdint>

template <int W>
class ap_int {
  static_assert(W >= 1 && W <= 64, "shim ap_int supports 1..64 bits");
  std::int64_t value_;

  static std::int64_t sign_extend(std::int64_t x) {
    if (W >= 64) {
      return x;
    }
    const std::int64_t mask = (std::int64_t(1) << W) - 1;
    const std::int64_t sign = std::int64_t(1) << (W - 1);
    const std::int64_t truncated = x & mask;
    return (truncated ^ sign) - sign;
  }

 public:
  ap_int() : value_(0) {}
  ap_int(std::int64_t x) : value_(sign_extend(x)) {}  // NOLINT(runtime/explicit)

  ap_int &operator=(std::int64_t x) {
    value_ = sign_extend(x);
    return *this;
  }

  operator std::int64_t() const { return value_; }

  bool operator==(const ap_int &o) const { return value_ == o.value_; }
  bool operator!=(const ap_int &o) const { return value_ != o.value_; }
};

template <int W>
class ap_uint {
  static_assert(W >= 1 && W <= 64, "shim ap_uint supports 1..64 bits");
  std::uint64_t value_;

  static std::uint64_t truncate(std::uint64_t x) {
    if (W >= 64) {
      return x;
    }
    const std::uint64_t mask = (std::uint64_t(1) << W) - 1;
    return x & mask;
  }

 public:
  ap_uint() : value_(0) {}
  ap_uint(std::uint64_t x) : value_(truncate(x)) {}  // NOLINT(runtime/explicit)

  ap_uint &operator=(std::uint64_t x) {
    value_ = truncate(x);
    return *this;
  }

  operator std::uint64_t() const { return value_; }

  bool operator==(const ap_uint &o) const { return value_ == o.value_; }
  bool operator!=(const ap_uint &o) const { return value_ != o.value_; }
};

#endif  // SCPN_HLS_SHIM_AP_INT_H
