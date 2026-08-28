// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Python-owned zeroizing ML-DSA-65 key

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use zeroize::Zeroizing;

use crate::{
    ml_dsa_codec::PUBLIC_KEY_BYTES,
    ml_dsa_signing::{keygen, sign_internal},
};

/// ML-DSA-65 signing key whose native secret allocation is wiped on drop.
#[pyclass]
pub(crate) struct MlDsaSigningKey {
    secret_key: Option<Zeroizing<Vec<u8>>>,
    public_key: [u8; PUBLIC_KEY_BYTES],
}

#[pymethods]
impl MlDsaSigningKey {
    #[new]
    fn new(seed: Vec<u8>) -> PyResult<Self> {
        let seed = Zeroizing::new(seed);
        if seed.len() != 32 {
            return Err(PyValueError::new_err("seed must be 32 bytes"));
        }
        let seed_array = Zeroizing::new(
            seed.as_slice()
                .try_into()
                .expect("validated ML-DSA seed length"),
        );
        let (public_key, secret_key) = keygen(&seed_array);
        Ok(Self {
            secret_key: Some(Zeroizing::new(secret_key)),
            public_key,
        })
    }

    fn sign(&self, message: Vec<u8>, context: Vec<u8>) -> PyResult<Vec<u8>> {
        if context.len() > u8::MAX as usize {
            return Err(PyValueError::new_err("context must be at most 255 bytes"));
        }
        let secret_key = self
            .secret_key
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("signing key has been destroyed"))?;
        Ok(sign_internal(secret_key.as_ref(), &message, &context)
            .map_err(PyRuntimeError::new_err)?
            .to_vec())
    }

    fn public_key(&self) -> Vec<u8> {
        self.public_key.to_vec()
    }

    fn destroy(&mut self) {
        self.secret_key = None;
    }

    fn is_destroyed(&self) -> bool {
        self.secret_key.is_none()
    }
}
