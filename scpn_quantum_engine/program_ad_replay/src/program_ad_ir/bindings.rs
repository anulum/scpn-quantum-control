// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD PyO3 bindings

#[cfg(feature = "pyo3")]
include!("json_admission.rs");

#[cfg(feature = "pyo3")]
type NativeReplayFailureCell = std::rc::Rc<std::cell::RefCell<Option<PyErr>>>;

#[cfg(feature = "pyo3")]
thread_local! {
    static ACTIVE_NATIVE_FAILURE: std::cell::RefCell<Option<NativeReplayFailureCell>> =
        const { std::cell::RefCell::new(None) };
}

/// Restore the active Python exception owner after return or unwind.
#[cfg(feature = "pyo3")]
struct NativeReplayFailureScope {
    previous: Option<NativeReplayFailureCell>,
}

#[cfg(feature = "pyo3")]
impl NativeReplayFailureScope {
    fn enter() -> (Self, NativeReplayFailureCell) {
        let previous = ACTIVE_NATIVE_FAILURE.with(|active| active.borrow().clone());
        let failure = previous.clone().unwrap_or_else(|| {
            std::rc::Rc::new(std::cell::RefCell::new(None))
        });
        ACTIVE_NATIVE_FAILURE.with(|active| {
            active.replace(Some(std::rc::Rc::clone(&failure)));
        });
        (Self { previous }, failure)
    }
}

#[cfg(feature = "pyo3")]
impl Drop for NativeReplayFailureScope {
    fn drop(&mut self) {
        ACTIVE_NATIVE_FAILURE.with(|active| {
            active.replace(self.previous.take());
        });
    }
}

#[cfg(feature = "pyo3")]
fn with_native_replay_inputs<'py, F>(
    py: Python<'py>,
    serialization: &Bound<'py, PyAny>,
    inputs: Option<&Bound<'py, PyAny>>,
    replay: F,
) -> PyResult<Py<pyo3::types::PyString>>
where
    F: FnOnce(&str, &[f64]) -> PyResult<String>,
{
    let input_object = inputs
        .map(|value| value.clone().unbind())
        .unwrap_or_else(|| py.None());
    let admission_module = py.import("scpn_quantum_control.native_replay_admission")?;
    let memory_admission = admission_module.getattr("_admit_native_replay_workspace")?.unbind();
    let manager = admission_module
        .getattr("native_replay_input_scope")?
        .call1((serialization, input_object))?;
    let reservation = manager.call_method0("__enter__")?;
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> PyResult<Py<pyo3::types::PyString>> {
        let input_bytes = reservation.getattr("decision")?.getattr("bytes_required")?.extract::<usize>()?;
        let native_reservation = reservation.clone().unbind();
        let memory_reservation = reservation.clone().unbind();
        let (_failure_owner, checkpoint_failure) = NativeReplayFailureScope::enter();
        let captured_failure = std::rc::Rc::clone(&checkpoint_failure);
        let memory_failure = std::rc::Rc::clone(&checkpoint_failure);
        let declared = std::cell::Cell::new(crate::program_ad_lifecycle::ReplayMemoryRequest::default());
        let admission = std::rc::Rc::new(move |request: crate::program_ad_lifecycle::ReplayMemoryRequest| {
                if memory_failure.borrow().is_some() {
                    return Err("native replay memory admission refused".to_owned());
                }
                let total = declared.get().checked_add(request)?;
                declared.set(total);
                let admitted = Python::attach(|memory_py| {
                    memory_admission.bind(memory_py).call1((
                        memory_reservation.bind(memory_py),
                        input_bytes,
                        total.forward_bytes,
                        total.adjoint_bytes,
                        total.intermediate_bytes,
                    )).map(|_| ())
                });
                match admitted {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let mut first_failure = memory_failure.borrow_mut();
                        if first_failure.is_none() {
                            *first_failure = Some(error);
                        }
                        Err("native replay memory admission refused".to_owned())
                    }
                }
            });
        let numeric_admission = std::rc::Rc::clone(&admission);
        let metadata_admission = std::rc::Rc::clone(&admission);
        let mut result = crate::program_ad_lifecycle::with_replay_metadata_admission(
            move |bytes| metadata_admission(crate::program_ad_lifecycle::ReplayMemoryRequest {
                intermediate_bytes: bytes,
                ..Default::default()
            }),
            || crate::program_ad_lifecycle::with_replay_memory_admission(
                move |request| numeric_admission(request),
            || {
                crate::program_ad_lifecycle::with_replay_checkpoint(
                    move || {
                        if captured_failure.borrow().is_some() {
                            return Err("native replay lifecycle checkpoint refused".to_owned());
                        }
                        let checked = Python::attach(|checkpoint_py| {
                            native_reservation.bind(checkpoint_py).call_method0("checkpoint")
                                .map(|_| ())
                        });
                        match checked {
                            Ok(()) => Ok(()),
                            Err(error) => {
                                let mut first_failure = captured_failure.borrow_mut();
                                if first_failure.is_none() {
                                    *first_failure = Some(error);
                                }
                                Err("native replay lifecycle checkpoint refused".to_owned())
                            }
                        }
                    },
                    || {
                        crate::program_ad_lifecycle::replay_checkpoint().map_err(PyValueError::new_err)?;
                        let source = serialization.extract::<String>()?;
                        let values = match inputs {
                            Some(value) => value.extract::<Vec<f64>>()?,
                            None => Vec::new(),
                        };
                        crate::program_ad_lifecycle::replay_checkpoint().map_err(PyValueError::new_err)?;
                        let encoded = replay(&source, &values)?;
                        crate::program_ad_lifecycle::replay_checkpoint().map_err(PyValueError::new_err)?;
                        let python_bytes = admission_module
                            .getattr("_native_replay_python_string_bytes")?
                            .call1((encoded.len(),))?
                            .extract::<usize>()?;
                        crate::program_ad_lifecycle::admit_replay_memory(
                            crate::program_ad_lifecycle::ReplayMemoryRequest {
                                forward_bytes: 0,
                                adjoint_bytes: 0,
                                intermediate_bytes: python_bytes,
                            },
                        ).map_err(PyValueError::new_err)?;
                        let output = pyo3::types::PyString::from_bytes(py, encoded.as_bytes())?.unbind();
                        crate::program_ad_lifecycle::replay_checkpoint().map_err(PyValueError::new_err)?;
                        Ok(output)
                    },
                )
            },
        ));
        let original_failure = checkpoint_failure.borrow().as_ref().map(|error| error.clone_ref(py));
        if let Some(error) = original_failure {
            result = Err(error);
        }
        result
    }));
    let result = match attempted {
        Ok(result) => result,
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native Program AD replay panicked; request refused",
        )),
    };
    let cleanup = match &result {
        Ok(_) => manager.call_method1("__exit__", (py.None(), py.None(), py.None())),
        Err(error) => manager.call_method1(
            "__exit__", (error.get_type(py), error.value(py), error.traceback(py)),
        ),
    };
    if let Err(cleanup_error) = cleanup {
        if let Err(original_error) = result {
            cleanup_error.set_cause(py, Some(original_error));
        }
        return Err(cleanup_error);
    }
    result
}

/// PyO3 wrapper returning a JSON metadata summary for a Program AD IR payload.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_metadata_summary<'py>(
    py: Python<'py>,
    serialization: &Bound<'py, PyAny>,
) -> PyResult<Py<pyo3::types::PyString>> {
    with_native_replay_inputs(py, serialization, None, |source, _| {
        let ir = parse_program_ad_effect_ir(source).map_err(PyValueError::new_err)?;
        encode_admitted_native_json(&ir.metadata_summary(), "Program AD IR summary")
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD interpretation.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_interpret_forward<'py>(
    py: Python<'py>,
    serialization: &Bound<'py, PyAny>,
    inputs: &Bound<'py, PyAny>,
) -> PyResult<Py<pyo3::types::PyString>> {
    with_native_replay_inputs(py, serialization, Some(inputs), |source, values| {
        let result = interpret_program_ad_effect_ir_forward(source, values)
            .map_err(PyValueError::new_err)?;
        encode_admitted_native_json(&result, "Program AD IR interpreter result")
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD value+gradient replay.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_interpret_value_and_gradient<'py>(
    py: Python<'py>,
    serialization: &Bound<'py, PyAny>,
    inputs: &Bound<'py, PyAny>,
) -> PyResult<Py<pyo3::types::PyString>> {
    with_native_replay_inputs(py, serialization, Some(inputs), |source, values| {
        let result = interpret_program_ad_effect_ir_value_and_gradient(source, values)
            .map_err(PyValueError::new_err)?;
        encode_admitted_native_json(&result, "Program AD IR value+gradient result")
    })
}
