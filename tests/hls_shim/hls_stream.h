// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//
// NON-SYNTHESIS lint / co-simulation shim for <hls_stream.h>. Backs
// hls::stream<T> with a std::deque FIFO so the generated AXI4-Stream pulse
// player runs as host software. Vivado/Vitis HLS supplies the authentic header
// at synthesis time.
#ifndef SCPN_HLS_SHIM_HLS_STREAM_H
#define SCPN_HLS_SHIM_HLS_STREAM_H

#include <cstddef>
#include <deque>
#include <stdexcept>

namespace hls {

template <typename T>
class stream {
  std::deque<T> fifo_;

 public:
  stream() = default;

  void write(const T &value) { fifo_.push_back(value); }

  bool write_nb(const T &value) {
    fifo_.push_back(value);
    return true;
  }

  T read() {
    if (fifo_.empty()) {
      throw std::runtime_error("hls::stream shim: read from empty stream");
    }
    T value = fifo_.front();
    fifo_.pop_front();
    return value;
  }

  bool read_nb(T &value) {
    if (fifo_.empty()) {
      return false;
    }
    value = fifo_.front();
    fifo_.pop_front();
    return true;
  }

  bool empty() const { return fifo_.empty(); }
  std::size_t size() const { return fifo_.size(); }
};

}  // namespace hls

#endif  // SCPN_HLS_SHIM_HLS_STREAM_H
