// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD order-statistic metadata and ordering

/// Sort in place with fallible lifecycle checkpoints and no auxiliary allocation.
pub(crate) fn checked_order<T>(values: &mut [T], less: impl Fn(&T, &T) -> bool) -> Result<(), String> {
    replay_checkpoint()?;
    let size = values.len();
    for root in (0..size / 2).rev() {
        sift_order(values, root, size, &less)?;
    }
    for end in (1..values.len()).rev() {
        replay_checkpoint()?;
        values.swap(0, end);
        sift_order(values, 0, end, &less)?;
    }
    replay_checkpoint()?;
    Ok(())
}

fn sift_order<T>(values: &mut [T], mut root: usize, end: usize, less: &impl Fn(&T, &T) -> bool) -> Result<(), String> {
    while root < end / 2 {
        replay_checkpoint()?;
        let mut child = root.checked_mul(2).and_then(|v| v.checked_add(1))
            .ok_or_else(|| "order-statistic heap index overflowed".to_owned())?;
        let right = child.checked_add(1)
            .ok_or_else(|| "order-statistic heap index overflowed".to_owned())?;
        if right < end && less(&values[child], &values[right]) { child = right; }
        if !less(&values[root], &values[child]) { break; }
        values.swap(root, child);
        root = child;
    }
    Ok(())
}
