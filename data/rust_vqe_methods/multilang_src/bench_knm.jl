# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — K_nm Julia benchmark runner
using JSON
using Statistics
function build_knm(n)
    k = Array{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        k[i,j] = 0.45 * exp(-0.3 * abs((i-1) - (j-1)))
    end
    anchors = Dict((1,2)=>0.302, (2,3)=>0.201, (3,4)=>0.252, (4,5)=>0.154)
    for ((i,j), value) in anchors
        if i <= n && j <= n
            k[i,j] = value
            k[j,i] = value
        end
    end
    if n > 15
        k[1,16] = max(k[1,16], 0.05)
        k[16,1] = k[1,16]
    end
    if n > 6
        k[5,7] = max(k[5,7], 0.15)
        k[7,5] = k[5,7]
    end
    return k
end
ns = [4, 8, 16, 32, 64]
repeats = 300
rows = []
for n in ns
    vals = Float64[]
    for _ in 1:repeats
        t0 = time_ns()
        build_knm(n)
        push!(vals, (time_ns() - t0) / 1.0e6)
    end
    push!(rows, Dict("language"=>"julia", "n"=>n, "median_ms"=>median(vals), "status"=>"ok"))
end
println(JSON.json(rows))
