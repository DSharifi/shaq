//! Futex-backed waiting for the shared-memory queues.
//!
//! The futex word is a 64-bit atomic whose low 32 bits change on (or just
//! before) every wake. For the single-cursor queues it is the publication
//! cursor itself — every publish advances it. A multi-cursor queue (the
//! broadcast, which has one cursor per lane) instead points waiters at a
//! dedicated wake counter that a publish bumps only when a waiter is present
//! (see [`Waiters::bump_and_wake`]); there the caller's `check` is what gates
//! on real data, and the counter exists only to break a racing `FUTEX_WAIT`.
//!
//! # Why no wake is lost
//!
//! The producer advances the futex word (Release), then issues a SeqCst fence
//! and loads `waiters`, skipping the wake syscall if it is zero. A waiter
//! increments `waiters`, issues a SeqCst fence (in `register`), then rechecks
//! for data before sleeping. SeqCst fences are totally ordered, and a load
//! after the later fence must see a store made before the earlier one. So
//! either the producer's load sees the waiter and wakes it, or the waiter's
//! recheck sees the new data (or `FUTEX_WAIT` bounces with `EAGAIN`) and it
//! never sleeps. Without the fences both loads can be stale and a wake can be
//! lost.
//!
//! # Waiting on several words
//!
//! [`wait_any`] sleeps until any of several words changes (`futex_waitv`,
//! Linux 5.16+). A caller registers with each word's [`Waiters`] through
//! [`Waiters::add_waiter`], issues one [`registration_fence`] for the batch,
//! snapshots every word, and rechecks before sleeping; the argument above then
//! holds for each word. One `FUTEX_WAKE` marks at most one waiter per word, but
//! a thread sleeping on several words can be marked by the wakes for all of
//! them, so a count-limited wake can be absorbed by a thread that then uses
//! another word. Callers of [`wait_any`] must pass such wakes on.
//!
//! # The futex word
//!
//! Userspace accesses the word only as a 64-bit atomic; the kernel's 32-bit
//! `FUTEX_WAIT` compare reads its low half (byte offset 0 on little-endian, 4
//! on big-endian), which is 4-byte aligned and cannot tear. Never materialize
//! an `&AtomicU32` into it in Rust code — only a raw pointer passed to the
//! syscall. If the word advances by an exact multiple of 2^32 between snapshot
//! and compare the wait oversleeps; that is bounded by the timeout and
//! astronomically unlikely.

use crate::sync::atomic::{fence, AtomicUsize, Ordering};
use crate::{error::WaitError, CacheAlignedAtomicSize};
use core::hint::spin_loop;
use std::time::{Duration, Instant};

/// Snapshot of a queue's 64-bit publication cursor.
type SequenceNumber = usize;

/// `None` when the timeout overflows `Instant`: the wait is unbounded.
pub(crate) fn deadline_from_timeout(timeout: Duration) -> Option<Instant> {
    Instant::now().checked_add(timeout)
}

#[cfg(not(loom))]
fn remaining_until(deadline: Instant) -> Result<Duration, WaitError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or(WaitError::Timeout)
}

#[derive(Default)]
#[repr(C)]
pub(crate) struct Waiters {
    /// Approximate count of waiters registered against the queue's
    /// publication cursor.
    waiters: CacheAlignedAtomicSize,
}

impl Waiters {
    pub(crate) fn reset(&mut self) {
        self.waiters.reset();
    }

    /// Runs `check` until it returns a value or `timeout` elapses.
    ///
    /// `cursor` is the queue's publication cursor used as the futex word
    /// while sleeping; an advance of `cursor` must imply that `check` can
    /// observe new data.
    ///
    /// `spins` is how many times to re-`check` before the first sleep. A caller
    /// whose `check` is itself expensive (e.g. it scans many lanes) should pass
    /// a smaller count so the total spin work stays bounded; see
    /// [`SPIN_ATTEMPTS`] for the baseline used by a unit-cost check.
    pub(crate) fn wait_for<T>(
        &self,
        cursor: &AtomicUsize,
        spins: usize,
        timeout: Duration,
        mut check: impl FnMut() -> Option<T>,
    ) -> Result<T, WaitError> {
        if let Some(value) = check() {
            return Ok(value);
        }

        // Taken only after the first check so an immediately satisfied call
        // never reads the clock; the timeout still bounds the spin below.
        let deadline = deadline_from_timeout(timeout);

        // Spin only before the first sleep; once this thread has blocked it
        // is on the slow path and goes straight back to waiting.
        for _ in 0..spins {
            spin_loop();
            if let Some(value) = check() {
                return Ok(value);
            }
        }

        loop {
            let snapshot = self.register(cursor);
            // Recheck after registering because a producer can publish after
            // the unregistered check and before this thread starts waiting;
            // the fence in `register` makes this recheck reliable (see
            // module docs).
            if let Some(value) = check() {
                self.unregister();
                return Ok(value);
            }

            // Platform waits can return after a matching wake or a spurious
            // wake, so success only means the caller's condition should be
            // checked again.
            let wait_result = Self::wait(cursor, snapshot, deadline);
            self.unregister();
            wait_result?;

            if let Some(value) = check() {
                return Ok(value);
            }
        }
    }

    /// Registers this thread as a waiter and snapshots the cursor.
    ///
    /// Increment-then-fence is the waiter half of the lost-wake protocol
    /// (see module docs); the caller must recheck for data after this call,
    /// before sleeping.
    fn register(&self, cursor: &AtomicUsize) -> SequenceNumber {
        self.add_waiter();
        registration_fence();
        cursor.load(Ordering::Acquire)
    }

    /// Counts this thread as a waiter without the fence, so a caller waiting
    /// on several words can register with each and then fence once. The
    /// caller must issue [`registration_fence`] before snapshotting the words.
    pub(crate) fn add_waiter(&self) {
        self.waiters.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn unregister(&self) {
        self.waiters.fetch_sub(1, Ordering::AcqRel);
    }

    fn wait(
        cursor: &AtomicUsize,
        expected: SequenceNumber,
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        // Avoid entering the platform wait backend if a publication already
        // advanced the cursor after the caller's post-registration recheck.
        if cursor.load(Ordering::Acquire) != expected {
            return Ok(());
        }

        imp::wait(cursor, expected, deadline)
    }

    /// Wakes up to `count` waiters registered against `cursor`.
    ///
    /// Callers must publish `cursor` with (at least) a Release store before
    /// calling this; the fence here pairs that store with a registering
    /// waiter (see module docs). The publication itself must be
    /// unconditional — only the wake syscall is elided when no waiter is
    /// registered.
    pub(crate) fn wake(&self, cursor: &AtomicUsize, count: usize) {
        debug_assert!(count > 0);

        // Fence-then-load is the producer half of the lost-wake protocol.
        // A plain load keeps the hot publish path free of RMWs on a line
        // consumers rarely write.
        fence(Ordering::SeqCst);
        let waiters = self.waiters.load(Ordering::Relaxed);
        if waiters == 0 {
            return;
        }

        let count = waiters.min(count).min(MAX_WAKE_COUNT) as u32;
        imp::wake(cursor, count);
    }

    /// Bumps `word` and wakes all registered waiters — but only if any are
    /// registered, so a queue with no blocked waiter never writes the shared
    /// `word` on its hot path.
    ///
    /// Unlike [`Self::wake`], the bump is conditional and done here: `word` is a
    /// dedicated wake counter that the caller does **not** otherwise advance, so
    /// this increment is what breaks a racing waiter's `FUTEX_WAIT`. Callers must
    /// have published the real data (the lane cursors a waiter rechecks) with a
    /// Release store before calling; the fence here pairs that with a registering
    /// waiter (see module docs).
    pub(crate) fn bump_and_wake(&self, word: &AtomicUsize) {
        fence(Ordering::SeqCst);
        let waiters = self.waiters.load(Ordering::Relaxed);
        if waiters == 0 {
            return;
        }

        // Change the futex word so a waiter that snapshotted it but has not yet
        // slept bounces out of `FUTEX_WAIT` with `EAGAIN`.
        word.fetch_add(1, Ordering::Relaxed);
        let count = waiters.min(MAX_WAKE_COUNT) as u32;
        imp::wake(word, count);
    }
}

const MAX_WAKE_COUNT: usize = i32::MAX as usize;

/// Baseline `check` attempts before a waiter's first sleep in
/// [`Waiters::wait_for`], for a unit-cost `check`. Callers with a costlier
/// `check` scale this down so the total spin work stays comparable.
#[cfg(not(loom))]
pub(crate) const SPIN_ATTEMPTS: usize = 2048;

/// Loom explores every interleaving of every spin, so models go straight to
/// registering.
#[cfg(loom)]
pub(crate) const SPIN_ATTEMPTS: usize = 0;

pub(crate) const MAX_WAIT_WORDS: usize = 32;

/// The waiter half of the lost-wake protocol for registrations made with
/// [`Waiters::add_waiter`]; see the module docs.
pub(crate) fn registration_fence() {
    fence(Ordering::SeqCst);
}

/// Blocks until any `(word, expected)` pair no longer matches, a wake arrives
/// for one of the words, or `deadline` passes.
///
/// `Ok(())` means the caller should recheck its condition: the return may be
/// spurious. The comparison uses each word's low 32 bits; see the module docs.
/// Where `futex_waitv` is unavailable, Linux sleeps on the first word alone,
/// for at most a millisecond at a time, and other platforms poll. Either way
/// the caller must register with every word's [`Waiters`], since a queue may
/// advance its word only when a waiter is registered.
pub(crate) fn wait_any(
    words: &[(&AtomicUsize, SequenceNumber)],
    deadline: Option<Instant>,
) -> Result<(), WaitError> {
    // The syscall's buffer holds at most `MAX_WAIT_WORDS` entries.
    assert!(!words.is_empty() && words.len() <= MAX_WAIT_WORDS);

    // Check the deadline first: words that keep moving must not hold off a
    // timeout forever.
    #[cfg(not(loom))]
    if let Some(deadline) = deadline {
        remaining_until(deadline)?;
    }

    // As in `Waiters::wait`, skip the syscall if a word already moved.
    if words
        .iter()
        .any(|&(word, expected)| word.load(Ordering::Acquire) != expected)
    {
        return Ok(());
    }

    match words {
        &[(word, expected)] => imp::wait(word, expected, deadline),
        words => imp::wait_any(words, deadline),
    }
}

#[cfg(all(target_os = "linux", not(loom)))]
mod imp {
    use super::{remaining_until, SequenceNumber, MAX_WAIT_WORDS};
    use crate::error::WaitError;
    use crate::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use core::mem::MaybeUninit;
    use std::time::{Duration, Instant};

    /// Cleared once `futex_waitv` is refused: by kernels before 5.16, or by a
    /// seccomp filter that does not allow it.
    static WAITV_SUPPORTED: AtomicBool = AtomicBool::new(true);

    /// How long a wait without `futex_waitv` sleeps on its first word before
    /// returning to recheck the others.
    const FALLBACK_INTERVAL: Duration = Duration::from_millis(1);

    /// Blocks with Linux `futex_waitv` while the low 32 bits of every word
    /// still equal those of its expected value. Where the syscall is refused,
    /// waits on the first word alone instead; see [`wait_first`].
    ///
    /// `Ok(())` means the caller should recheck its own condition.
    pub(super) fn wait_any(
        words: &[(&AtomicUsize, SequenceNumber)],
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        if !WAITV_SUPPORTED.load(Ordering::Relaxed) {
            return wait_first(words, deadline);
        }

        // SAFETY: an all-zero `futex_waitv` is valid, and its reserved field
        // must be zero.
        let mut vector: [libc::futex_waitv; MAX_WAIT_WORDS] = unsafe { core::mem::zeroed() };
        for (entry, &(word, expected)) in vector.iter_mut().zip(words) {
            entry.val = u64::from(expected as u32);
            entry.uaddr = futex_word(word) as u64;
            // No `FUTEX2_PRIVATE`: the words live in cross-process shared
            // memory.
            entry.flags = libc::FUTEX2_SIZE_U32 as u32;
        }

        loop {
            // Unlike `FUTEX_WAIT`, the timeout is an absolute time on the
            // given clock. A null timeout blocks indefinitely.
            let timeout_storage = match deadline {
                Some(deadline) => Some(monotonic_deadline(remaining_until(deadline)?)),
                None => None,
            };
            let timeout_ptr = timeout_storage
                .as_ref()
                .map_or(core::ptr::null(), |timeout| {
                    timeout as *const libc::timespec
                });

            // SAFETY:
            // - `vector` holds `words.len()` initialized entries, each naming
            //   a valid, 4-byte aligned pointer into a live 64-bit atomic;
            //   the kernel only reads 32 bits through it.
            // - `words.len()` is within `1..=MAX_WAIT_WORDS`, below the
            //   kernel's limit of 128, and the syscall's own flags are zero.
            // - `timeout_ptr` is null or points to a live `timespec`.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_futex_waitv,
                    vector.as_ptr(),
                    words.len() as libc::c_uint,
                    0 as libc::c_uint,
                    timeout_ptr,
                    libc::CLOCK_MONOTONIC,
                )
            };

            // On success the kernel returns the index of a woken word.
            if result >= 0 {
                return Ok(());
            }

            if result != -1 {
                panic!("unexpected futex_waitv result: {result}");
            }

            match errno() {
                libc::EAGAIN => return Ok(()),
                libc::EINTR => continue,
                libc::ETIMEDOUT => return Err(WaitError::Timeout),
                // The kernel could not allocate its copy of the vector.
                libc::ENOMEM => return wait_first(words, deadline),
                // `futex_waitv` itself never fails with `EPERM`, but a seccomp
                // filter that predates it may.
                libc::ENOSYS | libc::EPERM => {
                    WAITV_SUPPORTED.store(false, Ordering::Relaxed);
                    return wait_first(words, deadline);
                }
                err => panic!("unexpected futex_waitv error: errno={err}"),
            }
        }
    }

    /// Waits with `FUTEX_WAIT` on the first word alone, for at most
    /// [`FALLBACK_INTERVAL`], so the caller notices a change to any other
    /// word within that interval rather than by spinning.
    fn wait_first(
        words: &[(&AtomicUsize, SequenceNumber)],
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        let (word, expected) = words[0];
        let interval_end = Instant::now().checked_add(FALLBACK_INTERVAL);
        match (deadline, interval_end) {
            (Some(deadline), Some(interval_end)) if deadline <= interval_end => {
                wait(word, expected, Some(deadline))
            }
            (_, interval_end) => match wait(word, expected, interval_end) {
                // The interval, not the caller's deadline, ran out.
                Err(WaitError::Timeout) => Ok(()),
                result => result,
            },
        }
    }

    /// The absolute `CLOCK_MONOTONIC` time `remaining` from now, saturating.
    fn monotonic_deadline(remaining: Duration) -> libc::timespec {
        let mut now = MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: `now` is valid for writing one `timespec`.
        let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, now.as_mut_ptr()) };
        assert_eq!(result, 0, "CLOCK_MONOTONIC is always available");
        // SAFETY: `clock_gettime` succeeded, so it initialized `now`.
        let now = unsafe { now.assume_init() };

        // Both parts are below one second, so the sum fits in a `u32`.
        let mut nanos = now.tv_nsec as u32 + remaining.subsec_nanos();
        let mut secs = remaining.as_secs();
        if nanos >= 1_000_000_000 {
            nanos -= 1_000_000_000;
            secs = secs.saturating_add(1);
        }
        let tv_sec = (now.tv_sec as u64)
            .saturating_add(secs)
            .min(libc::time_t::MAX as u64) as libc::time_t;
        libc::timespec {
            tv_sec,
            tv_nsec: nanos as libc::c_long,
        }
    }

    /// Returns the futex word: the low 32 bits of the 64-bit cursor, the
    /// half that changes on every publication.
    ///
    /// Only the kernel reads through this pointer; never materialize an
    /// `&AtomicU32` over the cursor in Rust code (see module docs).
    fn futex_word(cursor: &AtomicUsize) -> *mut u32 {
        let ptr = cursor.as_ptr().cast();
        // The low half lives at byte offset 4 on big-endian targets.
        #[cfg(target_endian = "big")]
        // SAFETY: in bounds; the low half of the 8-byte cursor is at offset 4.
        let ptr = unsafe { ptr.add(1) };
        ptr
    }

    /// Blocks with Linux `FUTEX_WAIT` while the low 32 bits of `cursor` still
    /// equal the low 32 bits of `expected`.
    ///
    /// `Ok(())` means the caller should recheck its own condition; Linux can
    /// return success for ordinary wakes and for spurious wakes.
    pub(super) fn wait(
        cursor: &AtomicUsize,
        expected: SequenceNumber,
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        let expected = expected as u32;
        loop {
            // A null timeout makes `FUTEX_WAIT` block indefinitely.
            let timeout_storage = match deadline {
                Some(deadline) => Some(duration_to_timespec(remaining_until(deadline)?)),
                None => None,
            };
            let timeout_ptr = timeout_storage
                .as_ref()
                .map_or(core::ptr::null(), |timeout| {
                    timeout as *const libc::timespec
                });

            // SAFETY:
            // - `futex_word(cursor)` is a valid, 4-byte aligned pointer into a
            //   live 64-bit atomic; the kernel only reads 32 bits through it.
            // - `timeout_ptr` is null (block indefinitely) or points to a
            //   live `timespec`.
            // - `FUTEX_WAIT` only blocks if the value still equals `expected`.
            // - No `FUTEX_PRIVATE_FLAG`: the cursor lives in cross-process
            //   shared memory.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    futex_word(cursor),
                    libc::FUTEX_WAIT,
                    expected as libc::c_int,
                    timeout_ptr,
                )
            };

            if result == 0 {
                return Ok(());
            }

            if result != -1 {
                panic!("unexpected futex wait result: {result}");
            }

            match errno() {
                libc::EAGAIN => return Ok(()),
                libc::EINTR => continue,
                libc::ETIMEDOUT => return Err(WaitError::Timeout),
                err => panic!("unexpected futex wait error: errno={err}"),
            }
        }
    }

    /// Wakes waiters blocked in Linux `FUTEX_WAIT` on `cursor`.
    pub(super) fn wake(cursor: &AtomicUsize, count: u32) {
        debug_assert!(count <= libc::c_int::MAX as u32);

        // SAFETY:
        // - `futex_word(cursor)` is a valid, 4-byte aligned pointer into a
        //   live 64-bit atomic.
        // - No `FUTEX_PRIVATE_FLAG`: the cursor lives in cross-process
        //   shared memory.
        let result = unsafe {
            libc::syscall(
                libc::SYS_futex,
                futex_word(cursor),
                libc::FUTEX_WAKE,
                count as libc::c_int,
            )
        };
        debug_assert!(
            result >= 0,
            "unexpected futex wake error: errno={}",
            errno()
        );
    }

    /// Converts a [`Duration`] to the relative [`libc::timespec`]
    /// timeout format expected by futex.
    #[inline]
    const fn duration_to_timespec(duration: Duration) -> libc::timespec {
        // Clamp instead of casting: a wrapped-negative `tv_sec` would make
        // the kernel reject the timespec with EINVAL.
        let secs = duration.as_secs();
        let tv_sec = if secs > libc::time_t::MAX as u64 {
            libc::time_t::MAX
        } else {
            secs as libc::time_t
        };
        libc::timespec {
            tv_sec,
            tv_nsec: duration.subsec_nanos() as libc::c_long,
        }
    }

    fn errno() -> i32 {
        // SAFETY: `__errno_location` returns this thread's errno location on Linux.
        let errno = unsafe { libc::__errno_location() };
        // SAFETY: the returned pointer is valid and points to this thread's errno.
        unsafe { *errno }
    }
}

#[cfg(all(not(target_os = "linux"), not(loom)))]
mod imp {
    use super::{remaining_until, SequenceNumber};
    use crate::error::WaitError;
    use crate::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    /// Polls until any word no longer equals its expected value or `deadline`
    /// passes, yielding the thread between checks.
    pub(super) fn wait_any(
        words: &[(&AtomicUsize, SequenceNumber)],
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        loop {
            if words
                .iter()
                .any(|&(word, expected)| word.load(Ordering::Acquire) != expected)
            {
                return Ok(());
            }

            if let Some(deadline) = deadline {
                remaining_until(deadline)?;
            }

            std::thread::yield_now();
        }
    }

    /// Polls until `cursor` no longer equals `expected` or timeout elapses,
    /// yielding the thread between checks.
    ///
    /// `Ok(())` means the caller should recheck its own condition.
    pub(super) fn wait(
        cursor: &AtomicUsize,
        expected: SequenceNumber,
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        loop {
            if cursor.load(Ordering::Acquire) != expected {
                return Ok(());
            }

            if let Some(deadline) = deadline {
                remaining_until(deadline)?;
            }

            std::thread::yield_now();
        }
    }

    /// No-ops because spin waiters observe the shared cursor directly.
    pub(super) fn wake(_cursor: &AtomicUsize, _count: u32) {}
}

// Model the kernel's atomic compare-and-sleep operation. Timeouts and spurious
// wakes are omitted; threads remain asleep until explicitly woken.
#[cfg(loom)]
mod imp {
    use super::SequenceNumber;
    use crate::{
        error::WaitError,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use loom::{sync::Mutex, thread};
    use std::time::Instant;

    loom::lazy_static! {
        static ref SLEEPERS: Mutex<Vec<(usize, thread::Thread)>> = Mutex::new(Vec::new());
    }

    pub(super) fn wait(
        cursor: &AtomicUsize,
        expected: SequenceNumber,
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        assert!(
            deadline.is_none(),
            "Loom futex model does not model timeouts"
        );
        sleep_on(&[(cursor, expected)]);
        Ok(())
    }

    pub(super) fn wait_any(
        words: &[(&AtomicUsize, SequenceNumber)],
        deadline: Option<Instant>,
    ) -> Result<(), WaitError> {
        assert!(
            deadline.is_none(),
            "Loom futex model does not model timeouts"
        );
        sleep_on(words);
        Ok(())
    }

    /// Like `futex_waitv`, compares and queues one word at a time, so a wake
    /// can mark an early word before a later compare fails; that wake is then
    /// absorbed. Sleeps until a wake removes one of this thread's entries.
    ///
    /// Whether a wake arrived is read from the queue, not from park tokens:
    /// loom drops an unpark that reaches a thread blocked on the lock.
    fn sleep_on(words: &[(&AtomicUsize, SequenceNumber)]) {
        let current = thread::current();
        let mut queued = 0;
        for &(word, expected) in words {
            let mut sleepers = SLEEPERS.lock().unwrap();
            if word.load(Ordering::Relaxed) as u32 != expected as u32 {
                dequeue(&mut sleepers, &current);
                return;
            }
            sleepers.push((address(word), current.clone()));
            queued += 1;
        }

        loop {
            {
                let mut sleepers = SLEEPERS.lock().unwrap();
                let still_queued = sleepers
                    .iter()
                    .filter(|(_, queued)| queued.id() == current.id())
                    .count();
                if still_queued < queued {
                    // Like the kernel, never return with an entry queued.
                    dequeue(&mut sleepers, &current);
                    return;
                }
            }
            // An unpark between the check and here is kept as a token.
            thread::park();
        }
    }

    fn address(word: &AtomicUsize) -> usize {
        word as *const AtomicUsize as usize
    }

    fn dequeue(sleepers: &mut Vec<(usize, thread::Thread)>, sleeper: &thread::Thread) {
        sleepers.retain(|(_, queued)| queued.id() != sleeper.id());
    }

    pub(super) fn wake(cursor: &AtomicUsize, mut count: u32) {
        let address = address(cursor);
        let mut sleepers = SLEEPERS.lock().unwrap();
        let mut index = 0;
        while index < sleepers.len() && count > 0 {
            if sleepers[index].0 == address {
                let (_, sleeper) = sleepers.swap_remove(index);
                sleeper.unpark();
                count -= 1;
            } else {
                index += 1;
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn wait_any_times_out_although_a_word_moved() {
        let moved = AtomicUsize::new(1);
        let still = AtomicUsize::new(0);
        let words = [(&still, 0), (&moved, 0)];
        // A word that keeps moving must not hold off a deadline that passed.
        assert!(matches!(
            wait_any(&words, Some(Instant::now())),
            Err(WaitError::Timeout)
        ));
        assert!(wait_any(&words, None).is_ok());
    }
}
