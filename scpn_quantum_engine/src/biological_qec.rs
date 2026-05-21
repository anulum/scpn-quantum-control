// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Biological Surface-Code Decoder

//! Rust acceleration for biological-surface-code Z-error decoding.
//!
//! The routine accepts a weighted biological coupling graph (edge list + edge
//! weights) and an X-syndrome vector, then computes a minimum-weight perfect
//! matching correction using:
//! 1. Dijkstra shortest paths between defect pairs
//! 2. Exact MWPM via dynamic programming on bitmasks

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Copy, Clone)]
struct HeapState {
    cost: f64,
    node: usize,
}

impl Eq for HeapState {}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct DijkstraResult {
    dist: Vec<f64>,
    prev_node: Vec<Option<usize>>,
    prev_edge: Vec<Option<usize>>,
}

fn build_adjacency(
    n_nodes: usize,
    edge_u: &[i64],
    edge_v: &[i64],
    edge_w: &[f64],
) -> PyResult<Vec<Vec<(usize, usize, f64)>>> {
    let mut adj = vec![Vec::new(); n_nodes];
    for idx in 0..edge_u.len() {
        let u = edge_u[idx];
        let v = edge_v[idx];
        let w = edge_w[idx];
        if u < 0 || v < 0 {
            return Err(PyValueError::new_err(
                "edge endpoints must be non-negative.",
            ));
        }
        let uu = u as usize;
        let vv = v as usize;
        if uu >= n_nodes || vv >= n_nodes {
            return Err(PyValueError::new_err(format!(
                "edge ({uu}, {vv}) out of bounds for n_nodes={n_nodes}"
            )));
        }
        if !w.is_finite() || w <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "edge_weight[{idx}] must be finite and > 0, got {w}"
            )));
        }
        adj[uu].push((vv, idx, w));
        adj[vv].push((uu, idx, w));
    }
    Ok(adj)
}

fn dijkstra_from_source(source: usize, adj: &[Vec<(usize, usize, f64)>]) -> DijkstraResult {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev_node = vec![None; n];
    let mut prev_edge = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[source] = 0.0;
    heap.push(HeapState {
        cost: 0.0,
        node: source,
    });

    while let Some(HeapState { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        for &(next, edge_idx, weight) in &adj[node] {
            let next_cost = cost + weight;
            if next_cost < dist[next] {
                dist[next] = next_cost;
                prev_node[next] = Some(node);
                prev_edge[next] = Some(edge_idx);
                heap.push(HeapState {
                    cost: next_cost,
                    node: next,
                });
            }
        }
    }

    DijkstraResult {
        dist,
        prev_node,
        prev_edge,
    }
}

fn defect_components_have_even_parity(defects: &[usize], adj: &[Vec<(usize, usize, f64)>]) -> bool {
    let n = adj.len();
    let mut component = vec![usize::MAX; n];
    let mut comp_id = 0usize;
    for node in 0..n {
        if component[node] != usize::MAX {
            continue;
        }
        let mut stack = vec![node];
        component[node] = comp_id;
        while let Some(cur) = stack.pop() {
            for &(nxt, _, _) in &adj[cur] {
                if component[nxt] == usize::MAX {
                    component[nxt] = comp_id;
                    stack.push(nxt);
                }
            }
        }
        comp_id += 1;
    }

    let mut parity = vec![0u8; comp_id];
    for &defect in defects {
        parity[component[defect]] ^= 1;
    }
    parity.into_iter().all(|x| x == 0)
}

fn mwpm_exact(dist_mat: &[Vec<f64>]) -> Option<Vec<(usize, usize)>> {
    let n = dist_mat.len();
    if n % 2 != 0 {
        return None;
    }
    if n == 0 {
        return Some(Vec::new());
    }
    if n > 24 {
        return None;
    }

    let max_mask = 1usize << n;
    let mut dp = vec![f64::INFINITY; max_mask];
    let mut choice: Vec<Option<(usize, usize)>> = vec![None; max_mask];
    dp[0] = 0.0;

    for mask in 1..max_mask {
        if mask.count_ones() % 2 == 1 {
            continue;
        }
        let first = mask.trailing_zeros() as usize;
        let mask_without_first = mask & !(1usize << first);
        let mut j_mask = mask_without_first;
        while j_mask != 0 {
            let j = j_mask.trailing_zeros() as usize;
            let pair_mask = mask_without_first & !(1usize << j);
            let d = dist_mat[first][j];
            if d.is_finite() {
                let candidate = dp[pair_mask] + d;
                if candidate < dp[mask] {
                    dp[mask] = candidate;
                    choice[mask] = Some((first, j));
                }
            }
            j_mask &= j_mask - 1;
        }
    }

    let full = max_mask - 1;
    if !dp[full].is_finite() {
        return None;
    }

    let mut out = Vec::new();
    let mut mask = full;
    while mask != 0 {
        let (i, j) = choice[mask]?;
        out.push((i, j));
        mask &= !(1usize << i);
        mask &= !(1usize << j);
    }
    Some(out)
}

fn reconstruct_path_edges(source: usize, target: usize, dj: &DijkstraResult) -> Option<Vec<usize>> {
    if !dj.dist[target].is_finite() {
        return None;
    }
    let mut node = target;
    let mut edges = Vec::new();
    while node != source {
        let edge = dj.prev_edge[node]?;
        let prev = dj.prev_node[node]?;
        edges.push(edge);
        node = prev;
    }
    Some(edges)
}

#[pyfunction]
pub fn biological_decode_z_errors<'py>(
    py: Python<'py>,
    edge_u: PyReadonlyArray1<'_, i64>,
    edge_v: PyReadonlyArray1<'_, i64>,
    edge_weight: PyReadonlyArray1<'_, f64>,
    n_nodes: usize,
    syndrome_x: PyReadonlyArray1<'_, i8>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let u = edge_u.as_slice()?;
    let v = edge_v.as_slice()?;
    let w = edge_weight.as_slice()?;
    let syndrome = syndrome_x.as_slice()?;

    if n_nodes == 0 {
        return Err(PyValueError::new_err("n_nodes must be positive."));
    }
    if u.len() != v.len() || u.len() != w.len() {
        return Err(PyValueError::new_err(
            "edge_u, edge_v, and edge_weight must have the same length.",
        ));
    }
    if syndrome.len() != n_nodes {
        return Err(PyValueError::new_err(format!(
            "syndrome_x length {} does not match n_nodes {}",
            syndrome.len(),
            n_nodes
        )));
    }
    if syndrome.iter().any(|&x| x != 0 && x != 1) {
        return Err(PyValueError::new_err("syndrome_x must be binary (0/1)."));
    }

    let correction =
        biological_decode_inner(u, v, w, n_nodes, syndrome).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, correction))
}

/// Pure-Rust biological surface-code decoder used by wrapper and benchmarks.
pub fn biological_decode_inner(
    edge_u: &[i64],
    edge_v: &[i64],
    edge_weight: &[f64],
    n_nodes: usize,
    syndrome_x: &[i8],
) -> Result<Vec<i8>, String> {
    let adj = build_adjacency(n_nodes, edge_u, edge_v, edge_weight).map_err(|e| e.to_string())?;
    let defects: Vec<usize> = syndrome_x
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
        .collect();

    let mut correction = vec![0i8; edge_u.len()];
    if defects.is_empty() {
        return Ok(correction);
    }
    if defects.len() % 2 != 0 {
        return Err("syndrome_x contains odd number of defects.".to_owned());
    }
    if !defect_components_have_even_parity(&defects, &adj) {
        return Err("syndrome_x has odd syndrome parity in a connected component.".to_owned());
    }

    let dijkstra_per_defect: Vec<DijkstraResult> = defects
        .iter()
        .map(|&source| dijkstra_from_source(source, &adj))
        .collect();

    let n_defects = defects.len();
    let mut dist_mat = vec![vec![f64::INFINITY; n_defects]; n_defects];
    for i in 0..n_defects {
        dist_mat[i][i] = 0.0;
        for j in (i + 1)..n_defects {
            let d = dijkstra_per_defect[i].dist[defects[j]];
            dist_mat[i][j] = d;
            dist_mat[j][i] = d;
        }
    }

    let pairs = mwpm_exact(&dist_mat).ok_or_else(|| {
        "defects cannot be perfectly matched with finite shortest paths (or defect count exceeds exact-MWPM limit)."
            .to_owned()
    })?;

    for (i, j) in pairs {
        let source = defects[i];
        let target = defects[j];
        let path_edges = reconstruct_path_edges(source, target, &dijkstra_per_defect[i])
            .ok_or_else(|| "failed to reconstruct shortest path for matched pair.".to_owned())?;
        for edge_idx in path_edges {
            correction[edge_idx] ^= 1;
        }
    }
    Ok(correction)
}

#[cfg(test)]
mod tests {
    use super::{biological_decode_inner, mwpm_exact};

    #[test]
    fn test_mwpm_exact_simple_square() {
        let dist = vec![
            vec![0.0, 1.0, 2.0, 1.0],
            vec![1.0, 0.0, 1.0, 2.0],
            vec![2.0, 1.0, 0.0, 1.0],
            vec![1.0, 2.0, 1.0, 0.0],
        ];
        let pairs = mwpm_exact(&dist).expect("must match");
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_biological_decode_inner_chain_single_error() {
        let edge_u = vec![0, 1, 2];
        let edge_v = vec![1, 2, 3];
        let edge_w = vec![1.0, 1.0, 1.0];
        let syndrome = vec![0, 1, 1, 0];
        let correction = biological_decode_inner(&edge_u, &edge_v, &edge_w, 4, &syndrome)
            .expect("decoder must succeed");
        assert_eq!(correction.len(), 3);
        assert_eq!(correction[1], 1);
    }
}
