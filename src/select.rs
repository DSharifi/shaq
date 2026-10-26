//! Waiting on several queues at once.
//!
//! See [`select!`](crate::select!) and [`select_biased!`](crate::select_biased!).
//! `recv(consumer)` takes an item from any consumer that implements
//! [`Receive`]: an [`spsc`], [`mpmc`], or [`broadcast`] consumer.
//!
//! # Exclusive and shared endpoints
//!
//! What matters to a select is not the kind of queue but whether the selecting
//! thread is the only one operating on its end of it:
//!
//! - An *exclusive* endpoint (an `spsc` or `broadcast` consumer) is only ever
//!   advanced by its owner, which a select borrows mutably. A publish wakes its
//!   only waiter, or every waiter at once.
//! - A *shared* endpoint (an `mpmc` consumer) competes with the other
//!   consumers of its queue. A select borrows it immutably, so it may appear in
//!   several arms, and a publish wakes no more of its waiters than it published
//!   items.
//!
//! # Blocking
//!
//! A blocked select spins briefly, then sleeps on the futex words of all its
//! queues at once (`futex_waitv`, Linux 5.16+). On Linux without it (before
//! 5.16, such as Ubuntu 22.04's default 5.15, or under a seccomp filter that
//! refuses it), a select over several queues sleeps on its first queue alone
//! and checks the others at least every millisecond. Off Linux it polls them,
//! yielding between checks, as the single-queue waits do; that keeps a core
//! busy while it waits.
//!
//! A select sleeping on several shared queues can absorb a wake meant for
//! another consumer of one of them and then take from a different one. A
//! select that finishes after sleeping therefore passes a wake on to each
//! other shared queue it slept on that still holds unclaimed items. See the
//! [design notes] for the full protocol and why it loses no wake.
//!
//! [`spsc`]: crate::spsc
//! [`mpmc`]: crate::mpmc
//! [`broadcast`]: crate::broadcast
//! [design notes]: https://github.com/anza-xyz/shaq/blob/main/docs/design/select.md

use crate::{
    error::WaitError,
    futex::{self, Waiters, MAX_WAIT_WORDS, SPIN_ATTEMPTS},
    sync::atomic::{fence, AtomicUsize, Ordering},
};
use core::{hint::spin_loop, iter::Chain, ops::Range};
use std::time::{Duration, Instant};

const MAX_ARMS: usize = MAX_WAIT_WORDS;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A consumer that `recv` in a [`select!`](crate::select!) can take items
/// from.
///
/// Implemented by [`spsc::Consumer`](crate::spsc::Consumer),
/// [`mpmc::Consumer`](crate::mpmc::Consumer), and
/// [`broadcast::Consumer`](crate::broadcast::Consumer). This trait is sealed.
pub trait Receive: sealed::Sealed {}

/// A queue endpoint that only its owner operates on, through `&mut`.
#[doc(hidden)]
pub trait ExclusiveEndpoint: sealed::Sealed {
    type Output;

    /// Must not write shared state when nothing is ready, since a blocked
    /// select calls it repeatedly.
    fn try_take(&mut self) -> Option<Self::Output>;

    /// Returns this endpoint along with the futex word a blocked select sleeps
    /// on while [`Self::try_take`] fails.
    ///
    /// # Safety
    /// The word lives in memory that `*self` keeps alive, so the caller must
    /// not replace or drop `*self` through the returned reference while it uses
    /// the word.
    unsafe fn split_wait_word(&mut self) -> (&mut Self, WaitWord<'_>);
}

/// A queue endpoint that competes with the other endpoints of its queue,
/// through `&`.
#[doc(hidden)]
pub trait SharedEndpoint: sealed::Sealed {
    type Output;

    /// Must not write shared state when nothing is ready, since a blocked
    /// select calls it repeatedly.
    fn try_take(&self) -> Option<Self::Output>;

    fn wait_word(&self) -> WaitWord<'_>;
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct WaitWord<'a> {
    waiters: &'a Waiters,
    word: &'a AtomicUsize,
    /// Consumer reservation cursor of a queue whose publish wakes no more
    /// waiters than it published items; `word` is then its publication cursor.
    /// `None` for a word whose wake cannot be stranded.
    reservation: Option<&'a AtomicUsize>,
    /// Relative cost of checking the operation, used to scale the spin budget.
    check_cost: usize,
}

impl<'a> WaitWord<'a> {
    /// A word whose wakes cannot be stranded: only one thread waits on it, or
    /// a wake reaches every waiter.
    pub(crate) fn new(waiters: &'a Waiters, word: &'a AtomicUsize) -> Self {
        Self {
            waiters,
            word,
            reservation: None,
            check_cost: 1,
        }
    }

    /// The publication cursor of a queue shared by competing consumers, where
    /// a publish wakes at most as many waiters as it published items.
    /// Unclaimed items are those between `reservation` and `publication`.
    pub(crate) fn competing(
        waiters: &'a Waiters,
        publication: &'a AtomicUsize,
        reservation: &'a AtomicUsize,
    ) -> Self {
        Self {
            reservation: Some(reservation),
            ..Self::new(waiters, publication)
        }
    }

    pub(crate) fn with_check_cost(self, check_cost: usize) -> Self {
        Self { check_cost, ..self }
    }

    fn is(&self, other: &WaitWord<'_>) -> bool {
        core::ptr::eq(self.word, other.word)
    }

    /// Whether a wake this select absorbed but did not use could leave another
    /// consumer asleep beside unclaimed items.
    fn has_unclaimed_items(&self) -> bool {
        let Some(reservation) = self.reservation else {
            return false;
        };
        // Acquire: a consumer's claim implies the publications it saw, so the
        // publication load below sees them too, and a later item is not
        // mistaken for the claimed one.
        let reservation = reservation.load(Ordering::Acquire);
        self.word.load(Ordering::Acquire) != reservation
    }
}

#[doc(hidden)]
pub trait Operation<'q> {
    type Output;

    /// Writes no shared queue state when the operation is not ready.
    fn attempt(&mut self) -> Option<Self::Output>;

    fn word(&self) -> WaitWord<'q>;
}

#[doc(hidden)]
pub struct Exclusive<'a, E> {
    endpoint: &'a mut E,
    word: WaitWord<'a>,
}

impl<'a, E: ExclusiveEndpoint> Operation<'a> for Exclusive<'a, E> {
    type Output = E::Output;

    #[inline]
    fn attempt(&mut self) -> Option<Self::Output> {
        self.endpoint.try_take()
    }

    #[inline]
    fn word(&self) -> WaitWord<'a> {
        self.word
    }
}

#[doc(hidden)]
pub struct Shared<'a, E> {
    endpoint: &'a E,
    word: WaitWord<'a>,
}

impl<'a, E: SharedEndpoint> Operation<'a> for Shared<'a, E> {
    type Output = E::Output;

    #[inline]
    fn attempt(&mut self) -> Option<Self::Output> {
        self.endpoint.try_take()
    }

    #[inline]
    fn word(&self) -> WaitWord<'a> {
        self.word
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No attempt pass has failed yet.
    Start,
    /// Repeating attempt passes before registering.
    Spin,
    /// The next failed pass registers with the words.
    Register,
    /// Registered with and snapshotted the words; the failed pass was the
    /// recheck that decides whether to sleep on them.
    Recheck,
}

struct WordSet<'q, const N: usize> {
    words: [Option<WaitWord<'q>>; N],
    snapshots: [usize; N],
    len: usize,
}

impl<'q, const N: usize> WordSet<'q, N> {
    fn new() -> Self {
        Self {
            words: [None; N],
            snapshots: [0; N],
            len: 0,
        }
    }

    fn iter(&self) -> impl Iterator<Item = &WaitWord<'q>> {
        self.words[..self.len].iter().flatten()
    }

    fn insert(&mut self, word: WaitWord<'q>) {
        if !self.iter().any(|member| member.is(&word)) {
            self.words[self.len] = Some(word);
            self.len += 1;
        }
    }
}

/// Drives one `select!`: the order arms are tried in, and what to do between
/// attempt passes that find no arm ready. `N` is the number of arms.
///
/// After each failed pass the caller calls [`Self::wait`], which spins,
/// registers with the arms' words, or sleeps on them, and asks for another
/// pass. A blocked select only sleeps after a pass made while registered with
/// and holding snapshots of every word; see the module docs.
#[doc(hidden)]
pub struct Selector<'q, const N: usize> {
    arms: [WaitWord<'q>; N],
    timeout: Option<Duration>,
    start: usize,
    phase: Phase,
    spins: usize,
    deadline: Option<Instant>,
    /// The words to sleep on, collected from `arms` on the first block.
    words: WordSet<'q, N>,
    /// Whether this select is counted among the waiters of `words`.
    registered: bool,
    /// Whether this select slept on `words` since it last handed off wakes,
    /// so it may have absorbed some.
    slept: bool,
    timed_out: bool,
}

impl<'q, const N: usize> Selector<'q, N> {
    const ARMS: () = assert!(N > 0 && N <= MAX_ARMS);

    /// `timeout` of `None` blocks until an arm is ready; zero never blocks. A
    /// biased select tries arms in order, others from a random start.
    #[inline]
    pub fn new(arms: [WaitWord<'q>; N], timeout: Option<Duration>, biased: bool) -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::ARMS;
        let start = if biased || N <= 1 { 0 } else { random_index(N) };
        Self {
            arms,
            timeout,
            start,
            phase: Phase::Start,
            spins: 0,
            deadline: None,
            words: WordSet::new(),
            registered: false,
            slept: false,
            timed_out: false,
        }
    }

    /// Arm indices in the order to try them.
    #[inline]
    pub fn order(&self) -> Chain<Range<usize>, Range<usize>> {
        (self.start..N).chain(0..self.start)
    }

    /// Called after a pass in which no arm fired. Returns `false` once the
    /// select should give up and run its `default` arm; otherwise the caller
    /// must make another pass over every arm before calling this again.
    pub fn wait(&mut self) -> bool {
        if self.phase == Phase::Start {
            // Read the clock only on the first miss, so a select that finds a
            // ready arm never does.
            match self.timeout {
                Some(timeout) if timeout.is_zero() => return false,
                Some(timeout) => self.deadline = futex::deadline_from_timeout(timeout),
                None => {}
            }
            // Scale the spin budget down by what a pass costs, so a select
            // over costly checks spins about as long as a single-queue wait.
            let cost: usize = self.arms.iter().map(|word| word.check_cost).sum();
            self.spins = SPIN_ATTEMPTS / cost.max(1);
            self.phase = Phase::Spin;
        }

        // The pass after a timed-out sleep was the last.
        if self.timed_out {
            self.give_up();
            return false;
        }

        match self.phase {
            Phase::Spin if self.spins > 0 => {
                self.spins -= 1;
                spin_loop();
            }
            Phase::Recheck => self.sleep(),
            _ => self.register(),
        }
        true
    }

    /// Called once arm `index` took its output.
    #[inline]
    pub fn finish(&mut self, index: usize) {
        self.unregister();
        if self.slept {
            self.hand_off_wakes(Some(self.arms[index]));
        }
    }

    fn give_up(&mut self) {
        self.unregister();
        self.hand_off_wakes(None);
    }

    /// Registers with every word, then snapshots them, so the caller's next
    /// pass is the recheck the `futex` module's protocol requires before
    /// sleeping.
    fn register(&mut self) {
        if self.words.len == 0 {
            for word in self.arms {
                self.words.insert(word);
            }
        }

        // Even a select that polls instead of sleeping registers: a queue may
        // advance its word only while it has waiters.
        for word in self.words.iter() {
            word.waiters.add_waiter();
        }
        futex::registration_fence();
        self.registered = true;
        for index in 0..self.words.len {
            if let Some(word) = self.words.words[index] {
                self.words.snapshots[index] = word.word.load(Ordering::Acquire);
            }
        }
        self.phase = Phase::Recheck;
    }

    fn sleep(&mut self) {
        self.slept = true;
        let first = self.words.words[0].expect("a blocked select waits on a word");
        let pairs: [(&AtomicUsize, usize); N] = core::array::from_fn(|index| {
            let word = self.words.words[index].unwrap_or(first);
            (word.word, self.words.snapshots[index])
        });
        let result = futex::wait_any(&pairs[..self.words.len], self.deadline);
        self.unregister();
        // Order the next pass after whatever woke this select: a waker's own
        // fence preceded its read of the registration it woke us for.
        fence(Ordering::SeqCst);

        if let Err(WaitError::Timeout) = result {
            self.timed_out = true;
        }
        self.phase = Phase::Register;
    }

    fn unregister(&mut self) {
        if core::mem::take(&mut self.registered) {
            for word in self.words.iter() {
                word.waiters.unregister();
            }
        }
    }

    /// Re-issues a wake on every shared queue slept on, other than `consumed`,
    /// that still holds items, in case this select absorbed the wake for them.
    /// Spurious when no wake was absorbed, which only costs the woken consumer
    /// a recheck.
    fn hand_off_wakes(&mut self, consumed: Option<WaitWord<'q>>) {
        if !core::mem::take(&mut self.slept) {
            return;
        }
        for word in self.words.iter() {
            if consumed.is_some_and(|consumed| consumed.is(word)) {
                continue;
            }
            if word.has_unclaimed_items() {
                word.waiters.wake(word.word, 1);
            }
        }
    }
}

impl<const N: usize> Drop for Selector<'_, N> {
    fn drop(&mut self) {
        // Only reached registered if an attempt panicked.
        self.unregister();
    }
}

/// A pseudo-random index below `len`, so `select!` does not favor its first
/// arm when several are ready.
#[cfg(not(loom))]
fn random_index(len: usize) -> usize {
    use std::cell::Cell;

    std::thread_local! {
        static STATE: Cell<u32> = const { Cell::new(0x9E37_79B9) };
    }

    STATE
        .try_with(|state| {
            // xorshift32
            let mut x = state.get();
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            state.set(x);
            ((u64::from(x) * len as u64) >> 32) as usize
        })
        .unwrap_or(0)
}

/// Loom models must replay deterministically.
#[cfg(loom)]
fn random_index(_len: usize) -> usize {
    0
}

/// Support for the `select!` expansion. Not public API.
#[doc(hidden)]
pub mod __private {
    use super::{Exclusive, ExclusiveEndpoint, Receive, Shared, SharedEndpoint};
    use std::time::Duration;

    pub use super::{Operation, Selector};

    /// Borrows a shared endpoint operand through method-call autoref, so an
    /// operation accepts `rx`, `&rx`, `&mut rx`, or a reference binding.
    pub trait BindShared {
        fn __shaq_select_bind(&self) -> &Self;
    }

    impl<E: SharedEndpoint + Receive> BindShared for E {
        #[inline]
        fn __shaq_select_bind(&self) -> &Self {
            self
        }
    }

    /// Borrows an exclusive endpoint operand through method-call autoref, so
    /// an operation accepts `rx` (a `mut` binding), `&mut rx`, or a `&mut`
    /// binding, which is reborrowed rather than moved.
    pub trait BindExclusive {
        fn __shaq_select_bind(&mut self) -> &mut Self;
    }

    impl<E: ExclusiveEndpoint + Receive> BindExclusive for E {
        #[inline]
        fn __shaq_select_bind(&mut self) -> &mut Self {
            self
        }
    }

    /// The operand traits, for the expansion to glob-import: the import adds no
    /// names, and it is used by every select, so it needs no `allow`.
    pub mod bind {
        pub use super::{BindExclusive as _, BindShared as _};
    }

    pub trait IntoRecv {
        type Op;

        fn into_recv(self) -> Self::Op;
    }

    impl<'a, E: SharedEndpoint + Receive> IntoRecv for &'a E {
        type Op = Shared<'a, E>;

        #[inline]
        fn into_recv(self) -> Self::Op {
            Shared {
                endpoint: self,
                word: self.wait_word(),
            }
        }
    }

    impl<'a, E: ExclusiveEndpoint + Receive> IntoRecv for &'a mut E {
        type Op = Exclusive<'a, E>;

        #[inline]
        fn into_recv(self) -> Self::Op {
            // SAFETY: `Exclusive` only takes from the endpoint; it never
            // replaces or drops it.
            let (endpoint, word) = unsafe { self.split_wait_word() };
            Exclusive { endpoint, word }
        }
    }

    #[inline]
    pub fn recv<O: IntoRecv>(operand: O) -> O::Op {
        operand.into_recv()
    }

    /// `select! { default(timeout) => .. }` with no other arms.
    pub fn sleep(timeout: Duration) {
        if !timeout.is_zero() {
            std::thread::sleep(timeout);
        }
    }
}

/// Waits on several queues and runs the first arm whose operation succeeds,
/// like `crossbeam_channel::select!`.
///
/// ```text
/// select! {
///     recv(rx) -> pattern => body,
///     ...
///     default => body,           // optional: give up if nothing is ready
///     default(timeout) => body,  // or: give up after `timeout` (a `Duration`)
/// }
/// ```
///
/// `recv(rx)` takes an item from an [`spsc`], [`mpmc`], or [`broadcast`]
/// consumer (see [`Receive`]) and binds it to the arm's pattern. Nothing is
/// taken from an arm that does not run.
///
/// Each operand is a place such as `rx` or `self.rx`, or a reference such as
/// `&mut rx`. The `default` timeout is evaluated first, then each operand once,
/// in order. An [`mpmc`] consumer may also be given as `&rx`, and may appear
/// in several arms; any other consumer is borrowed mutably, so it may appear
/// in only one. The select runs exactly one arm's body. When several arms are
/// ready, the select tries them from a pseudo-random arm onwards (use
/// [`select_biased!`](crate::select_biased!) to prefer earlier arms). With no
/// `default` arm the select blocks until an arm is ready; `default` gives up
/// immediately and `default(timeout)` after `timeout`. The whole select is an
/// expression whose value is that of the body that ran.
///
/// Unlike `crossbeam_channel`, a receive yields the item itself rather than a
/// `Result`: shaq queues have no disconnection to report, so with no `default`
/// arm a select over queues no producer will write to blocks forever.
///
/// Bodies run after the select is complete, so `break`, `continue`, `return`,
/// and `?` apply to the surrounding code, and a body may use the operands
/// again. A body that is not a block, such as an `if` or a `match`, must be
/// followed by a comma. A select has at least one arm and at most 32 `recv`
/// arms, plus at most one `default`. See the [`select`](mod@crate::select)
/// module for how a blocked select waits.
///
/// [`spsc`]: crate::spsc
/// [`mpmc`]: crate::mpmc
/// [`broadcast`]: crate::broadcast
/// [`Receive`]: crate::select::Receive
///
/// # Examples
///
/// ```
/// use shaq::{mpmc, select, spsc};
/// use std::time::Duration;
///
/// let (events_tx, events) = mpmc::pair::<u32>(16).unwrap();
/// let (mut jobs_tx, mut jobs) = spsc::pair::<u64>(16).unwrap();
///
/// jobs_tx.try_write(7).unwrap();
/// let received = select! {
///     recv(events) -> event => format!("event {event}"),
///     recv(jobs) -> job => format!("job {job}"),
///     default(Duration::from_secs(1)) => panic!("nothing arrived"),
/// };
/// assert_eq!(received, "job 7");
///
/// // A plain `default` never blocks.
/// let received = select! {
///     recv(events) -> event => Some(event),
///     recv(jobs) -> job => Some(job as u32),
///     default => None,
/// };
/// assert_eq!(received, None);
/// # drop(events_tx);
/// ```
///
/// Bodies may break out of, or continue, the surrounding loop:
///
/// ```
/// use shaq::{mpmc, select};
///
/// let (data_tx, data) = mpmc::pair::<u32>(16).unwrap();
/// let (stop_tx, stop) = mpmc::pair::<u8>(1).unwrap();
/// for value in 1..=3 {
///     data_tx.try_write(value).unwrap();
/// }
///
/// let mut sum = 0;
/// loop {
///     select! {
///         recv(data) -> value => {
///             sum += value;
///             if sum == 6 {
///                 stop_tx.try_write(0).unwrap();
///             }
///         }
///         recv(&stop) -> _ => break,
///     }
/// }
/// assert_eq!(sum, 6);
/// ```
#[macro_export]
macro_rules! select {
    ($($tokens:tt)*) => {
        $crate::__select_internal!(@list ($($tokens)*) () () false)
    };
}

/// Like [`select!`](crate::select!), but when several arms are ready it runs
/// the first one listed instead of a random one.
///
/// A queue listed early can starve later ones if it is never empty.
///
/// ```
/// use shaq::{mpmc, select_biased};
///
/// let (urgent_tx, urgent) = mpmc::pair::<&str>(4).unwrap();
/// let (normal_tx, normal) = mpmc::pair::<&str>(4).unwrap();
/// normal_tx.try_write("normal").unwrap();
/// urgent_tx.try_write("urgent").unwrap();
///
/// let first = select_biased! {
///     recv(urgent) -> message => message,
///     recv(normal) -> message => message,
/// };
/// assert_eq!(first, "urgent");
/// ```
#[macro_export]
macro_rules! select_biased {
    ($($tokens:tt)*) => {
        $crate::__select_internal!(@list ($($tokens)*) () () true)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __select_internal {
    // Parse the arms into `[op [operand] [pattern] [body]]`, and `default`
    // into `[timeout] [body]`.
    (@list () $arms:tt $default:tt $biased:tt) => {
        $crate::__select_internal!(
            @assign $arms ()
            (
                (0 _r0 _m0) (1 _r1 _m1) (2 _r2 _m2) (3 _r3 _m3)
                (4 _r4 _m4) (5 _r5 _m5) (6 _r6 _m6) (7 _r7 _m7)
                (8 _r8 _m8) (9 _r9 _m9) (10 _r10 _m10) (11 _r11 _m11)
                (12 _r12 _m12) (13 _r13 _m13) (14 _r14 _m14) (15 _r15 _m15)
                (16 _r16 _m16) (17 _r17 _m17) (18 _r18 _m18) (19 _r19 _m19)
                (20 _r20 _m20) (21 _r21 _m21) (22 _r22 _m22) (23 _r23 _m23)
                (24 _r24 _m24) (25 _r25 _m25) (26 _r26 _m26) (27 _r27 _m27)
                (28 _r28 _m28) (29 _r29 _m29) (30 _r30 _m30) (31 _r31 _m31)
            )
            $default $biased
        )
    };
    (@list (default => $($tail:tt)*) $arms:tt () $biased:tt) => {
        $crate::__select_internal!(@default ($($tail)*) [::core::time::Duration::ZERO] $arms $biased)
    };
    (@list (default($t:expr) => $($tail:tt)*) $arms:tt () $biased:tt) => {
        $crate::__select_internal!(@default ($($tail)*) [$t] $arms $biased)
    };
    (@list (default $($tail:tt)*) $arms:tt ($($default:tt)+) $biased:tt) => {
        ::core::compile_error!("a `select!` may have only one `default` arm")
    };
    // The last arm; its trailing comma is optional.
    (@list
        ($op:ident($o:expr) -> $p:pat => $body:expr $(,)?)
        ($($arms:tt)*) $default:tt $biased:tt
    ) => {
        $crate::__select_internal!(
            @list () ($($arms)* [$op [$o] [$p] [$body]]) $default $biased
        )
    };
    (@list
        ($op:ident($o:expr) -> $p:pat => $body:expr, $($tail:tt)*)
        ($($arms:tt)*) $default:tt $biased:tt
    ) => {
        $crate::__select_internal!(
            @list ($($tail)*) ($($arms)* [$op [$o] [$p] [$body]]) $default $biased
        )
    };
    // A block body needs no comma.
    (@list
        ($op:ident($o:expr) -> $p:pat => $body:block $($tail:tt)*)
        ($($arms:tt)*) $default:tt $biased:tt
    ) => {
        $crate::__select_internal!(
            @list ($($tail)*) ($($arms)* [$op [$o] [$p] [$body]]) $default $biased
        )
    };
    (@list ($($tokens:tt)*) $arms:tt $default:tt $biased:tt) => {
        ::core::compile_error!(
            "expected `recv(operand) -> pattern => body`, `default => body`, \
             or `default(timeout) => body`"
        )
    };

    (@default ($body:expr $(,)?) $t:tt $arms:tt $biased:tt) => {
        $crate::__select_internal!(@list () $arms ($t [$body]) $biased)
    };
    (@default ($body:expr, $($tail:tt)*) $t:tt $arms:tt $biased:tt) => {
        $crate::__select_internal!(@list ($($tail)*) $arms ($t [$body]) $biased)
    };
    (@default ($body:block $($tail:tt)*) $t:tt $arms:tt $biased:tt) => {
        $crate::__select_internal!(@list ($($tail)*) $arms ($t [$body]) $biased)
    };

    // Give each arm an index and hygienic names for its operation and output.
    (@assign () $arms:tt $names:tt $default:tt $biased:tt) => {
        $crate::__select_internal!(@emit $arms $default $biased)
    };
    (@assign ($arm:tt $($rest:tt)*) ($($arms:tt)*) ($name:tt $($names:tt)*) $default:tt $biased:tt) => {
        $crate::__select_internal!(
            @assign ($($rest)*) ($($arms)* [$name $arm]) ($($names)*) $default $biased
        )
    };
    (@assign $rest:tt $arms:tt () $default:tt $biased:tt) => {
        ::core::compile_error!("a `select!` supports at most 32 arms")
    };

    (@op recv $o:expr) => {
        $crate::select::__private::recv($o.__shaq_select_bind())
    };
    (@op $op:ident $o:expr) => {
        ::core::compile_error!(::core::concat!(
            "unknown `select!` operation `",
            ::core::stringify!($op),
            "`; expected `recv`"
        ))
    };

    (@emit () () $biased:tt) => {
        ::core::compile_error!("a `select!` needs at least one arm")
    };
    (@emit () ([$t:expr] [$body:expr]) $biased:tt) => {{
        $crate::select::__private::sleep($t);
        // In a match arm, a braced body does not trip `unused_braces`.
        match () {
            () => $body,
        }
    }};
    (@emit $arms:tt () $biased:tt) => {
        $crate::__select_internal!(
            @select $arms (::core::option::Option::None) (::core::unreachable!()) $biased
        )
    };
    (@emit $arms:tt ([$t:expr] [$body:expr]) $biased:tt) => {
        $crate::__select_internal!(
            @select $arms (::core::option::Option::Some($t)) ($body) $biased
        )
    };

    (@select
        ($([($i:tt $r:ident $m:ident) [$op:ident [$o:expr] [$p:pat] [$body:expr]]])+)
        ($timeout:expr) ($otherwise:expr) $biased:tt
    ) => {{
        // Before the operands borrow anything it might use.
        let timeout: ::core::option::Option<::core::time::Duration> = $timeout;
        $( let mut $m = ::core::option::Option::None; )+
        // The operands stay borrowed only within this block, so the bodies
        // below may use them again.
        let selected: ::core::option::Option<usize> = {
            use $crate::select::__private::bind::*;
            $( let mut $r = $crate::__select_internal!(@op $op $o); )+
            let mut selector = $crate::select::__private::Selector::new(
                [$($crate::select::__private::Operation::word(&$r)),+],
                timeout,
                $biased,
            );
            'select: loop {
                for index in selector.order() {
                    match index {
                        $(
                            $i => if let ::core::option::Option::Some(output) =
                                $crate::select::__private::Operation::attempt(&mut $r)
                            {
                                selector.finish(index);
                                $m = ::core::option::Option::Some(output);
                                break 'select ::core::option::Option::Some(index);
                            },
                        )+
                        _ => ::core::unreachable!(),
                    }
                }
                if !selector.wait() {
                    break 'select ::core::option::Option::None;
                }
            }
        };
        match selected {
            // Binding in a match arm keeps a braced body from tripping
            // `unused_braces`, as it would in a block's tail position. The
            // inner match reports a refutable pattern against the item's type.
            $(
                ::core::option::Option::Some($i) => match $m {
                    ::core::option::Option::Some(output) => match output {
                        $p => $body,
                    },
                    ::core::option::Option::None => ::core::unreachable!(),
                },
            )+
            ::core::option::Option::None => $otherwise,
            ::core::option::Option::Some(_) => ::core::unreachable!(),
        }
    }};
}

#[cfg(all(test, loom))]
mod loom_tests;

#[cfg(all(test, not(loom)))]
mod tests {
    use crate::{mpmc, spsc};
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc,
        },
        thread,
        time::{Duration, Instant},
    };

    /// Bounds blocking selects so a lost wake fails the test instead of
    /// hanging it.
    const WATCHDOG: Duration = Duration::from_secs(10);

    /// Runs a select bounded by [`WATCHDOG`] and fails if it only completed
    /// through the final check after timing out, which hides a lost wake.
    #[cfg(not(miri))]
    fn before_watchdog<T>(select: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = select();
        assert!(start.elapsed() < WATCHDOG, "a wake was lost");
        value
    }

    /// Gives a select on another thread time to go to sleep.
    #[cfg(not(miri))]
    const SETTLE: Duration = Duration::from_millis(50);

    #[cfg(not(miri))]
    fn broadcast_pair() -> (
        crate::broadcast::Producer<u64>,
        crate::broadcast::Consumer<u64>,
    ) {
        use crate::broadcast::{Broadcast, BroadcastConfig};

        let file = crate::shmem::create_temp_shmem_file().unwrap();
        let config = BroadcastConfig {
            capacity: 8,
            producer_slots: 2,
            consumer_slots: 2,
        };
        // SAFETY: a fresh temp file initialized once; every handle uses `u64`.
        let queue = unsafe { Broadcast::<u64>::create(&file, config) }.unwrap();
        (queue.producer().unwrap(), queue.consumer().unwrap())
    }

    #[test]
    fn receives_from_spsc_consumers() {
        let (mut tx_a, mut rx_a) = spsc::pair::<u64>(4).unwrap();
        let (mut tx_b, mut rx_b) = spsc::pair::<u64>(4).unwrap();
        tx_b.try_write(2).unwrap();
        let received = crate::select! {
            recv(rx_a) -> value => (0, value),
            recv(rx_b) -> value => (1, value),
            default => panic!("an arm is ready"),
        };
        assert_eq!(received, (1, 2));

        tx_a.try_write(1).unwrap();
        let received = crate::select! {
            recv(&mut rx_a) -> value => value,
            recv(rx_b) -> value => value,
            default => panic!("an arm is ready"),
        };
        assert_eq!(received, 1);
        assert!(rx_a.try_read().is_none() && rx_b.try_read().is_none());
    }

    #[cfg(not(miri))]
    #[test]
    fn receives_from_broadcast_consumers() {
        let (mut tx, mut rx) = broadcast_pair();
        let (_mpmc_tx, mpmc_rx) = mpmc::pair::<u64>(4).unwrap();
        let ran_default = crate::select! {
            recv(rx) -> _ => false,
            recv(mpmc_rx) -> _ => false,
            default => true,
        };
        assert!(ran_default);

        tx.try_write(3).unwrap();
        let received = crate::select! {
            recv(rx) -> value => value,
            recv(mpmc_rx) -> value => value,
        };
        assert_eq!(received, 3);
    }

    #[cfg(not(miri))]
    #[test]
    fn blocks_until_any_kind_of_queue_publishes() {
        for late in 0..3 {
            let (mut spsc_tx, mut spsc_rx) = spsc::pair::<u64>(4).unwrap();
            let (mpmc_tx, mpmc_rx) = mpmc::pair::<u64>(4).unwrap();
            let (mut broadcast_tx, mut broadcast_rx) = broadcast_pair();

            let producer = thread::spawn(move || {
                thread::sleep(SETTLE);
                match late {
                    0 => spsc_tx.try_write(10).unwrap(),
                    1 => mpmc_tx.try_write(11).unwrap(),
                    _ => broadcast_tx.try_write(12).unwrap(),
                }
                (spsc_tx, mpmc_tx, broadcast_tx)
            });

            let received = before_watchdog(|| {
                crate::select! {
                    recv(spsc_rx) -> value => value,
                    recv(mpmc_rx) -> value => value,
                    recv(broadcast_rx) -> value => value,
                    default(WATCHDOG) => panic!("publish to arm {late} was missed"),
                }
            });
            assert_eq!(received, 10 + late);
            drop(producer.join().unwrap());
        }
    }

    #[test]
    fn takes_from_the_ready_arm_only() {
        let (tx_a, rx_a) = mpmc::pair::<u64>(4).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();

        tx_b.try_write(2).unwrap();
        let received = crate::select! {
            recv(rx_a) -> value => (0, value),
            recv(rx_b) -> value => (1, value),
            default => panic!("an arm is ready"),
        };
        assert_eq!(received, (1, 2));

        tx_a.try_write(1).unwrap();
        tx_b.try_write(3).unwrap();
        let first = crate::select! {
            recv(rx_a) -> value => value,
            recv(rx_b) -> value => value,
            default => panic!("both arms are ready"),
        };
        // The arm not taken keeps its value.
        let second = if first == 1 {
            rx_b.try_read()
        } else {
            rx_a.try_read()
        };
        assert_eq!(first + second.unwrap(), 4);
        assert!(rx_a.try_read().is_none());
        assert!(rx_b.try_read().is_none());
    }

    #[test]
    fn default_runs_when_nothing_is_ready() {
        let (_tx, rx) = mpmc::pair::<u64>(4).unwrap();
        let (_spsc_tx, mut spsc_rx) = spsc::pair::<u64>(4).unwrap();
        let ran_default = crate::select! {
            recv(rx) -> _ => false,
            recv(spsc_rx) -> _ => false,
            default => true,
        };
        assert!(ran_default);

        let ran_default = crate::select! {
            recv(rx) -> _ => false,
            default(Duration::ZERO) => true,
        };
        assert!(ran_default);
    }

    #[cfg(not(miri))]
    #[test]
    fn default_timeout_elapses_without_consuming() {
        let (_tx, rx) = mpmc::pair::<u64>(4).unwrap();
        let (_tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();

        let timeout = Duration::from_millis(20);
        let start = Instant::now();
        let timed_out = crate::select! {
            recv(rx) -> _ => false,
            recv(rx_b) -> _ => false,
            default(timeout) => true,
        };
        assert!(timed_out);
        assert!(start.elapsed() >= timeout);
    }

    #[cfg(not(miri))]
    #[test]
    fn blocks_until_any_queue_publishes() {
        for late in 0..3 {
            let (tx_a, rx_a) = mpmc::pair::<u64>(4).unwrap();
            let (tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();
            let (tx_c, rx_c) = mpmc::pair::<u64>(4).unwrap();

            let producer = thread::spawn(move || {
                thread::sleep(SETTLE);
                match late {
                    0 => tx_a.try_write(10).unwrap(),
                    1 => tx_b.try_write(11).unwrap(),
                    _ => tx_c.try_write(12).unwrap(),
                }
                (tx_a, tx_b, tx_c)
            });

            let received = before_watchdog(|| {
                crate::select! {
                    recv(rx_a) -> value => value,
                    recv(rx_b) -> value => value,
                    recv(rx_c) -> value => value,
                    default(WATCHDOG) => panic!("publish to arm {late} was missed"),
                }
            });
            assert_eq!(received, 10 + late);
            drop(producer.join().unwrap());
        }
    }

    #[test]
    fn select_picks_among_ready_arms() {
        let (tx_a, rx_a) = mpmc::pair::<u64>(4).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();
        let mut counts = [0; 2];
        for _ in 0..1000 {
            let _ = tx_a.try_write(0);
            let _ = tx_b.try_write(1);
            let index = crate::select! {
                recv(rx_a) -> value => value,
                recv(rx_b) -> value => value,
            };
            counts[index as usize] += 1;
        }
        assert!(counts.iter().all(|&count| count > 100), "{counts:?}");
    }

    #[test]
    fn select_biased_prefers_earlier_arms() {
        let (tx_a, rx_a) = mpmc::pair::<u64>(4).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();
        tx_b.try_write(1).unwrap();
        for _ in 0..100 {
            tx_a.try_write(0).unwrap();
            let index = crate::select_biased! {
                recv(rx_a) -> value => value,
                recv(rx_b) -> value => value,
            };
            assert_eq!(index, 0);
        }
        let index = crate::select_biased! {
            recv(rx_a) -> value => value,
            recv(rx_b) -> value => value,
        };
        assert_eq!(index, 1);
    }

    #[test]
    fn bodies_apply_control_flow_to_surrounding_code() {
        fn first_even(rx: &mpmc::Consumer<u64>) -> Result<u64, &'static str> {
            loop {
                crate::select! {
                    recv(rx) -> value => {
                        if value % 2 == 1 {
                            continue;
                        }
                        return Ok(value);
                    }
                    default => Err("empty")?,
                }
            }
        }

        let (tx, rx) = mpmc::pair::<u64>(8).unwrap();
        for value in [1, 3, 4, 5] {
            tx.try_write(value).unwrap();
        }
        assert_eq!(first_even(&rx), Ok(4));
        assert_eq!(rx.try_read(), Some(5));
        assert_eq!(first_even(&rx), Err("empty"));

        let mut seen = 0;
        tx.try_write(7).unwrap();
        loop {
            crate::select! {
                recv(rx) -> value => seen = value,
                default => break,
            }
        }
        assert_eq!(seen, 7);
    }

    #[test]
    fn bodies_may_reuse_the_operands() {
        let (tx, rx) = mpmc::pair::<u64>(8).unwrap();
        let (mut spsc_tx, mut spsc_rx) = spsc::pair::<u64>(8).unwrap();
        tx.try_write(1).unwrap();
        tx.try_write(2).unwrap();
        spsc_tx.try_write(3).unwrap();
        spsc_tx.try_write(4).unwrap();
        let drained = crate::select_biased! {
            recv(rx) -> first => first + rx.try_read().unwrap(),
            recv(spsc_rx) -> _ => unreachable!(),
        };
        assert_eq!(drained, 3);
        let drained = crate::select! {
            recv(rx) -> _ => unreachable!(),
            recv(spsc_rx) -> first => first + spsc_rx.try_read().unwrap(),
        };
        assert_eq!(drained, 7);
    }

    #[test]
    fn accepts_operand_places_and_references() {
        struct Endpoints {
            rx: mpmc::Consumer<u64>,
            spsc_rx: spsc::Consumer<u64>,
        }

        fn by_reference(rx: &mpmc::Consumer<u64>, spsc_rx: &mut spsc::Consumer<u64>) -> u64 {
            // A `&mut` binding is reborrowed, so it can be used repeatedly.
            let mut total = 0;
            for _ in 0..4 {
                total += crate::select! {
                    recv(rx) -> value => value,
                    recv(spsc_rx) -> value => value,
                    default => 0,
                };
            }
            total
        }

        let (tx, rx) = mpmc::pair::<u64>(8).unwrap();
        let (mut spsc_tx, spsc_rx) = spsc::pair::<u64>(8).unwrap();
        let mut endpoints = Endpoints { rx, spsc_rx };

        // Places: fields, and an `mpmc` consumer behind a non-`mut` binding.
        tx.try_write(1).unwrap();
        let shared_rx = endpoints.rx.clone();
        let value = crate::select! {
            recv(endpoints.spsc_rx) -> value => value,
            recv(shared_rx) -> value => value,
        };
        assert_eq!(value, 1);

        // Explicit references, including the same queue twice.
        spsc_tx.try_write(2).unwrap();
        let value = crate::select! {
            recv(&endpoints.rx) -> value => value,
            recv(&mut endpoints.spsc_rx) -> value => value,
            recv(&shared_rx) -> value => value,
        };
        assert_eq!(value, 2);

        tx.try_write(3).unwrap();
        spsc_tx.try_write(4).unwrap();
        assert_eq!(by_reference(&endpoints.rx, &mut endpoints.spsc_rx), 7);

        let mut boxed = Box::new(endpoints.spsc_rx);
        spsc_tx.try_write(5).unwrap();
        let value = crate::select! {
            recv(boxed) -> value => value,
        };
        assert_eq!(value, 5);
    }

    #[test]
    fn accepts_arm_syntax_variants() {
        let (tx, rx) = mpmc::pair::<(u64, u64)>(8).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<u64>(8).unwrap();

        tx.try_write((1, 2)).unwrap();
        // Block bodies with and without commas, a destructuring pattern, a
        // `default` before the arms, and no trailing comma.
        let sum = crate::select! {
            default => 0
            ,
            recv(rx_b) -> _ => { 0 }
            recv(rx) -> (a, b) => { a + b },
        };
        assert_eq!(sum, 3);

        tx_b.try_write(5).unwrap();
        let value = crate::select! {
            recv(rx_b) -> value => value
        };
        assert_eq!(value, 5);

        assert_eq!(crate::select! { default => 7 }, 7);
        assert_eq!(crate::select! { default(Duration::ZERO) => 8, }, 8);
    }

    #[test]
    fn supports_many_arms() {
        let queues: Vec<_> = (0..32).map(|_| mpmc::pair::<u64>(1).unwrap()).collect();
        let [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, q24, q25, q26, q27, q28, q29, q30, q31] =
            &queues[..]
        else {
            unreachable!()
        };
        q31.0.try_write(31).unwrap();
        let value = crate::select! {
            recv(q0.1) -> v => v, recv(q1.1) -> v => v, recv(q2.1) -> v => v,
            recv(q3.1) -> v => v, recv(q4.1) -> v => v, recv(q5.1) -> v => v,
            recv(q6.1) -> v => v, recv(q7.1) -> v => v, recv(q8.1) -> v => v,
            recv(q9.1) -> v => v, recv(q10.1) -> v => v, recv(q11.1) -> v => v,
            recv(q12.1) -> v => v, recv(q13.1) -> v => v, recv(q14.1) -> v => v,
            recv(q15.1) -> v => v, recv(q16.1) -> v => v, recv(q17.1) -> v => v,
            recv(q18.1) -> v => v, recv(q19.1) -> v => v, recv(q20.1) -> v => v,
            recv(q21.1) -> v => v, recv(q22.1) -> v => v, recv(q23.1) -> v => v,
            recv(q24.1) -> v => v, recv(q25.1) -> v => v, recv(q26.1) -> v => v,
            recv(q27.1) -> v => v, recv(q28.1) -> v => v, recv(q29.1) -> v => v,
            recv(q30.1) -> v => v, recv(q31.1) -> v => v,
            default => 0,
        };
        assert_eq!(value, 31);
    }

    /// Selects and plain blocking reads compete for `mpmc` items. Every item
    /// is consumed exactly once, and with no timeouts to recover from a lost
    /// wake, every consumer reaches its stop marker.
    #[cfg(not(miri))]
    #[test]
    fn competing_selects_and_reads_lose_no_wakes() {
        const ITEMS: u64 = 20_000;
        const STOP: u64 = u64::MAX;
        const SELECTORS: usize = 3;

        let (tx_a, rx_a) = mpmc::pair::<u64>(4).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<u64>(4).unwrap();
        let total = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = mpsc::channel();

        let mut consumers = Vec::new();
        for _ in 0..SELECTORS {
            let (rx_a, rx_b) = (rx_a.clone(), rx_b.clone());
            let (total, sum, done_tx) = (total.clone(), sum.clone(), done_tx.clone());
            consumers.push(thread::spawn(move || {
                loop {
                    let value = crate::select! {
                        recv(rx_a) -> value => value,
                        recv(rx_b) -> value => value,
                    };
                    if value == STOP {
                        break;
                    }
                    total.fetch_add(1, Ordering::Relaxed);
                    sum.fetch_add(value as usize, Ordering::Relaxed);
                }
                done_tx.send(()).unwrap();
            }));
        }
        // A plain consumer of `a` relies on `a`'s wakes reaching it.
        {
            let rx_a = rx_a.clone();
            let (total, sum, done_tx) = (total.clone(), sum.clone(), done_tx.clone());
            consumers.push(thread::spawn(move || {
                loop {
                    let value = rx_a.read_timeout(Duration::MAX).unwrap();
                    if value == STOP {
                        break;
                    }
                    total.fetch_add(1, Ordering::Relaxed);
                    sum.fetch_add(value as usize, Ordering::Relaxed);
                }
                done_tx.send(()).unwrap();
            }));
        }

        let producers: Vec<_> = [tx_a.clone(), tx_b]
            .into_iter()
            .enumerate()
            .map(|(lane, tx)| {
                thread::spawn(move || {
                    for value in (lane as u64..ITEMS).step_by(2) {
                        while tx.try_write(value).is_err() {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();
        for producer in producers {
            producer.join().unwrap();
        }
        // Each consumer takes at most one stop marker, and all of them read `a`.
        for _ in 0..consumers.len() {
            while tx_a.try_write(STOP).is_err() {
                thread::yield_now();
            }
        }

        for _ in 0..consumers.len() {
            done_rx
                .recv_timeout(WATCHDOG)
                .expect("a consumer missed a wake");
        }
        for consumer in consumers {
            consumer.join().unwrap();
        }
        assert_eq!(total.load(Ordering::Relaxed), ITEMS as usize);
        let expected = (ITEMS * (ITEMS - 1) / 2) as usize;
        assert_eq!(sum.load(Ordering::Relaxed), expected);
    }
}
