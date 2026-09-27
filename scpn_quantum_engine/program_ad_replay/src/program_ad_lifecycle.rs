// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — owned cooperative program-AD replay checkpoints

//! Owned, thread-local cooperative checkpoints for the shared pure replay.
//!
//! Nested scopes inherit their parent's policy. Standalone Rust/WASM replay
//! without a scope has no caller cancellation policy. Guards restore prior
//! ownership on return or unwind; they never stop another thread or forcibly
//! interrupt an opaque vendor call. Declared retained numeric memory can also
//! be presented to an owned admission policy before replay materialises values.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// Safety ceiling on iterations inside one opaque replay solver call.
///
/// This policy bound is not a wall-clock deadline or measured convergence claim.
/// Owners can tighten it with [`with_replay_solver_iterations`].
pub const DEFAULT_REPLAY_SOLVER_ITERATIONS: usize = 10_000;

type Checkpoint = Rc<dyn Fn() -> Result<(), String>>;
type MemoryAdmission = Rc<dyn Fn(ReplayMemoryRequest) -> Result<(), String>>;
type MetadataAdmission = Rc<dyn Fn(usize) -> Result<(), String>>;

thread_local! {
    static ACTIVE_CHECKPOINT: RefCell<Option<Checkpoint>> = const { RefCell::new(None) };
    static ACTIVE_SOLVER_ITERATIONS: Cell<usize> = const { Cell::new(DEFAULT_REPLAY_SOLVER_ITERATIONS) };
    static ACTIVE_MEMORY_ADMISSION: RefCell<Option<MemoryAdmission>> = const { RefCell::new(None) };
    static ACTIVE_METADATA_ADMISSION: RefCell<Option<MetadataAdmission>> = const { RefCell::new(None) };
}

struct CheckpointGuard {
    previous: Option<Checkpoint>,
}

impl Drop for CheckpointGuard {
    fn drop(&mut self) {
        ACTIVE_CHECKPOINT.with(|active| {
            active.replace(self.previous.take());
        });
    }
}

/// Run an operation under an owned cooperative checkpoint policy.
///
/// The callback owns its captured state and is invoked on this thread only.
/// Every child checkpoint first checks the active parent, so a nested caller
/// cannot clear cancellation or extend its parent's deadline. Restoration is
/// guaranteed when the operation returns or unwinds. Kernel callers must invoke
/// [`replay_checkpoint`] at allocation/iteration and native-operation boundaries.
pub fn with_replay_checkpoint<R>(
    checkpoint: impl Fn() -> Result<(), String> + 'static,
    operation: impl FnOnce() -> R,
) -> R {
    let previous = ACTIVE_CHECKPOINT.with(|active| active.borrow().clone());
    let inherited = previous.clone();
    let owned: Checkpoint = Rc::new(move || {
        if let Some(parent) = &inherited {
            parent()?;
        }
        checkpoint()
    });
    ACTIVE_CHECKPOINT.with(|active| {
        active.replace(Some(owned));
    });
    let _guard = CheckpointGuard { previous };
    operation()
}

/// Observe the current thread's owned policy, propagating its original refusal.
///
/// The callback is invoked after releasing the thread-local borrow, permitting
/// nested replay from a checkpoint without RefCell reentrancy panics. No active
/// policy means the standalone caller did not request cooperative interruption.
pub fn replay_checkpoint() -> Result<(), String> {
    let checkpoint = ACTIVE_CHECKPOINT.with(|active| active.borrow().clone());
    match checkpoint {
        Some(checkpoint) => checkpoint(),
        None => Ok(()),
    }
}

/// Declared numeric bytes presented before retained replay values are created.
///
/// The request is a declaration, not an allocator measurement. Kernel-local
/// workspace and metadata require their own declarations; zero does not mean
/// an unmeasured vendor operation allocates nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayMemoryRequest {
    /// Retained float64 primal values for this replay.
    pub forward_bytes: usize,
    /// Retained adjoints and the flattened parameter gradient.
    pub adjoint_bytes: usize,
    /// Additional explicitly declared intermediate bytes.
    pub intermediate_bytes: usize,
}

impl ReplayMemoryRequest {
    /// Sum every declared role, refusing overflow and native addressability.
    pub fn total_bytes(self) -> Result<usize, String> {
        self.forward_bytes
            .checked_add(self.adjoint_bytes)
            .and_then(|bytes| bytes.checked_add(self.intermediate_bytes))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                "Program AD retained replay bytes exceed native addressability".to_owned()
            })
    }

    /// Accumulate declarations without overflow before presenting a new total.
    ///
    /// An owner can conservatively retain all requests until its scope exits,
    /// including nested requests. This cumulative bound is not a live peak.
    pub fn checked_add(self, other: Self) -> Result<Self, String> {
        let add = |left: usize, right: usize| {
            left.checked_add(right).ok_or_else(|| {
                "Program AD retained replay bytes exceed native addressability".to_owned()
            })
        };
        let request = Self {
            forward_bytes: add(self.forward_bytes, other.forward_bytes)?,
            adjoint_bytes: add(self.adjoint_bytes, other.adjoint_bytes)?,
            intermediate_bytes: add(self.intermediate_bytes, other.intermediate_bytes)?,
        };
        request.total_bytes()?;
        Ok(request)
    }
}

struct MemoryAdmissionGuard {
    previous: Option<MemoryAdmission>,
}

impl Drop for MemoryAdmissionGuard {
    fn drop(&mut self) {
        ACTIVE_MEMORY_ADMISSION.with(|active| {
            active.replace(self.previous.take());
        });
    }
}

/// Run replay with an owned admission callback for declared numeric storage.
///
/// Every request first reaches the parent policy. A nested caller cannot waive
/// parent refusal. Callbacks receive additional declarations and own their
/// cumulative accounting; they are invoked outside the thread-local borrow.
/// The guard restores the prior policy on return or unwind. Standalone callers
/// without this policy get checked addressability, not host-budget admission.
pub fn with_replay_memory_admission<R>(
    admission: impl Fn(ReplayMemoryRequest) -> Result<(), String> + 'static,
    operation: impl FnOnce() -> R,
) -> R {
    let previous = ACTIVE_MEMORY_ADMISSION.with(|active| active.borrow().clone());
    let inherited = previous.clone();
    let owned: MemoryAdmission = Rc::new(move |request| {
        if let Some(parent) = &inherited {
            parent(request)?;
        }
        admission(request)
    });
    ACTIVE_MEMORY_ADMISSION.with(|active| {
        active.replace(Some(owned));
    });
    let _guard = MemoryAdmissionGuard { previous };
    operation()
}

pub(crate) fn admit_replay_memory(request: ReplayMemoryRequest) -> Result<(), String> {
    replay_checkpoint()?;
    request.total_bytes()?;
    let admission = ACTIVE_MEMORY_ADMISSION.with(|active| active.borrow().clone());
    match admission {
        Some(admission) => admission(request),
        None => Ok(()),
    }
}

struct MetadataAdmissionGuard {
    previous: Option<MetadataAdmission>,
}

impl Drop for MetadataAdmissionGuard {
    fn drop(&mut self) {
        ACTIVE_METADATA_ADMISSION.with(|active| active.replace(self.previous.take()));
    }
}

/// Run parsing under an owned admission policy for additional metadata bytes.
///
/// Metadata requests are separate from numerical replay declarations. Every
/// request reaches the parent first, outside thread-local borrows; restoration
/// is guaranteed on return or unwind. Callers own cumulative accounting.
/// Without a policy, only native addressability is checked.
pub fn with_replay_metadata_admission<R>(
    admission: impl Fn(usize) -> Result<(), String> + 'static,
    operation: impl FnOnce() -> R,
) -> R {
    let previous = ACTIVE_METADATA_ADMISSION.with(|active| active.borrow().clone());
    let inherited = previous.clone();
    let owned: MetadataAdmission = Rc::new(move |bytes| {
        if let Some(parent) = &inherited {
            parent(bytes)?;
        }
        admission(bytes)
    });
    ACTIVE_METADATA_ADMISSION.with(|active| active.replace(Some(owned)));
    let _guard = MetadataAdmissionGuard { previous };
    operation()
}

pub(crate) fn admit_replay_metadata(bytes: usize) -> Result<(), String> {
    replay_checkpoint()?;
    if bytes > isize::MAX as usize {
        return Err("Program AD parser metadata exceeds native addressability".to_owned());
    }
    let admission = ACTIVE_METADATA_ADMISSION.with(|active| active.borrow().clone());
    match admission {
        Some(admission) => admission(bytes),
        None => Ok(()),
    }
}

struct SolverIterationGuard {
    previous: usize,
}

impl Drop for SolverIterationGuard {
    fn drop(&mut self) {
        ACTIVE_SOLVER_ITERATIONS.with(|active| active.set(self.previous));
    }
}

/// Run an operation with a tighter positive opaque-solver iteration ceiling.
///
/// Zero is refused without entering `operation`, because the vendor treats zero
/// as unlimited. The effective ceiling is the minimum of this request and the
/// current parent's ceiling, initially [`DEFAULT_REPLAY_SOLVER_ITERATIONS`].
/// Nested operations cannot extend a parent's budget. Return or unwind restores
/// the previous ceiling on this thread. The bound applies per decomposition;
/// it does not interrupt a solver iteration or replace cooperative deadlines.
///
/// # Errors
///
/// Returns an error for a zero ceiling; otherwise returns the operation's result.
pub fn with_replay_solver_iterations<R>(
    max_iterations: usize,
    operation: impl FnOnce() -> R,
) -> Result<R, String> {
    if max_iterations == 0 {
        return Err("Program AD solver iteration limit must be positive".to_owned());
    }
    let previous = ACTIVE_SOLVER_ITERATIONS.with(|active| {
        let previous = active.get();
        active.set(previous.min(max_iterations));
        previous
    });
    let _guard = SolverIterationGuard { previous };
    Ok(operation())
}

pub(crate) fn replay_solver_iterations() -> usize {
    ACTIVE_SOLVER_ITERATIONS.with(Cell::get)
}
