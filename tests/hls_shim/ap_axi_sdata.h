// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//
// NON-SYNTHESIS lint / co-simulation shim for <ap_axi_sdata.h>. Mirrors the
// AMD Xilinx ap_axis<> side-channel layout (signed data plus TKEEP/TSTRB/TUSER/
// TLAST/TID/TDEST) with the shim ap_int/ap_uint backing, so a host build can
// drive and check the generated AXI4-Stream pulse player. Vivado/Vitis HLS
// supplies the authentic header at synthesis time.
#ifndef SCPN_HLS_SHIM_AP_AXI_SDATA_H
#define SCPN_HLS_SHIM_AP_AXI_SDATA_H

#include "ap_int.h"

namespace scpn_hls_shim_detail {
constexpr int byte_count(int data_width) { return (data_width + 7) / 8; }
constexpr int at_least_one(int width) { return width > 0 ? width : 1; }
}  // namespace scpn_hls_shim_detail

template <int Wdata, int Wuser, int Wid, int Wdest>
struct ap_axis {
  ap_int<Wdata> data;
  ap_uint<scpn_hls_shim_detail::byte_count(Wdata)> keep;
  ap_uint<scpn_hls_shim_detail::byte_count(Wdata)> strb;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wuser)> user;
  ap_uint<1> last;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wid)> id;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wdest)> dest;
};

template <int Wdata, int Wuser, int Wid, int Wdest>
struct ap_axiu {
  ap_uint<Wdata> data;
  ap_uint<scpn_hls_shim_detail::byte_count(Wdata)> keep;
  ap_uint<scpn_hls_shim_detail::byte_count(Wdata)> strb;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wuser)> user;
  ap_uint<1> last;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wid)> id;
  ap_uint<scpn_hls_shim_detail::at_least_one(Wdest)> dest;
};

#endif  // SCPN_HLS_SHIM_AP_AXI_SDATA_H
