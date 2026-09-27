// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Native result JSON buffer admission

struct NativeJsonWriter {
    length: usize,
    storage: Option<Vec<u8>>,
    limit: usize,
}

impl std::io::Write for NativeJsonWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        crate::program_ad_lifecycle::replay_checkpoint().map_err(std::io::Error::other)?;
        let length = self.length.checked_add(bytes.len())
            .filter(|length| *length <= self.limit)
            .ok_or_else(|| std::io::Error::other("native JSON exceeds admitted storage"))?;
        if let Some(storage) = &mut self.storage {
            storage.extend_from_slice(bytes);
        }
        self.length = length;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        crate::program_ad_lifecycle::replay_checkpoint().map_err(std::io::Error::other)
    }
}

fn encode_admitted_native_json<T: serde::Serialize>(value: &T, label: &str) -> PyResult<String> {
    let mut counter = NativeJsonWriter {length: 0, storage: None, limit: isize::MAX as usize};
    serde_json::to_writer(&mut counter, value).map_err(|error| {
        PyValueError::new_err(format!("failed to count {label}: {error}"))
    })?;
    crate::program_ad_lifecycle::admit_replay_memory(
        crate::program_ad_lifecycle::ReplayMemoryRequest {
            forward_bytes: 0,
            adjoint_bytes: 0,
            intermediate_bytes: counter.length,
        },
    ).map_err(PyValueError::new_err)?;
    let storage = reserve_replay_buffer(counter.length).map_err(PyValueError::new_err)?;
    let mut writer = NativeJsonWriter {length: 0, storage: Some(storage), limit: counter.length};
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        PyValueError::new_err(format!("failed to encode {label}: {error}"))
    })?;
    if writer.length != counter.length {
        return Err(PyValueError::new_err("native JSON encoded size changed after admission"));
    }
    let storage = writer.storage.ok_or_else(|| PyValueError::new_err("native JSON storage is missing"))?;
    String::from_utf8(storage).map_err(|error| {
        PyValueError::new_err(format!("native JSON encoding was not UTF-8: {error}"))
    })
}
