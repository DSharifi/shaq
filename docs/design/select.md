# `select!` over shaq queues

Status: step 1 of 3 (`recv`).
Tracking: anza-xyz/agave#16311, review thread anza-xyz/agave-block-production#47 (discussion_r4352959674).

## Motivation

The block-verification scheduler waits on several queues at once: replay messages to forward to workers, and
worker responses. Today it uses crossbeam's `select!`. To move those channels to shaq queues, shaq needs a
`select!` of its own.

The scheduler also forwards replay messages to the workers' `mpmc` queue, and must drain replay **only while**
that queue has room. Today it calls `has_room_for_request()`, then a `send()` that panics if called in the wrong
order. The issue asks for the type system to enforce this instead. That needs a select arm that waits for room,
which is what steps 2 and 3 add.

## Roadmap

| Step | Adds | Shared-memory ABI |
|---|---|---|
| **1 (this)** | `recv` from `spsc`, `mpmc` and `broadcast` consumers | unchanged |
| 2 | `send(tx, value)` on producers, so a select can wait for room | mpmc producers gain waiters |
| 3 | `reserve(tx) && recv(rx)`: fire only when *both* are ready, with a cancellable write permit | mpmc gains a single-producer mode |

With step 2, the scheduler holds at most one transformed request while the workers are full:

```rust
match pending {
    Some(_) => select! {
        send(worker_tx, pending.take().unwrap()) -> () => {}
        recv(response_rx) -> response => state.on_response(response),
    },
    None => select! {
        recv(replay_rx) -> message => *pending = state.to_request(message),
        recv(response_rx) -> response => state.on_response(response),
    },
}
```

Step 3 is needed only if even that one held message is a problem. A prototype exists; it is not part of this
change.

## API

### `select!` and `select_biased!`

```text
select! {
    recv(consumer) -> pattern => body,   // up to 32 arms
    ...
    default => body,              // optional: give up if nothing is ready
    default(timeout) => body,     // or: give up after `timeout`
}
```

- **Operation.** `recv(consumer)` takes an item and binds it to the pattern. It yields the item itself, not a
  `Result`: shaq queues have no disconnection to report.
- **Arm choice.** `select!` tries the arms from a pseudo-random one onwards. `select_biased!` tries them in written
  order.
- **Bodies** run after the select completes and has released its operands, so `break`, `continue`, `return` and
  `?` apply to the enclosing code, and a body may use the operands again.
- **Evaluation.** The `default` timeout is evaluated first, then each operand once, in order. An operand is accepted
  as a place (`rx`, `self.rx`), as a reference (`&mut rx`, or `&rx` for an mpmc consumer), or as a reference
  binding, which is reborrowed rather than moved, so it works in a loop.
- **Rejected at compile time:** an empty `select!`, `recv` on something that is not a consumer, an unknown
  operation, more than 32 arms, more than one `default`, and the same `spsc` or `broadcast` consumer twice
  (borrowck).

### Endpoints

`select` names no queue type. It defines two hidden endpoint traits, and each queue module implements one of them
for its consumer:

| Endpoint | Kind | Futex word | Implemented in |
|---|---|---|---|
| `spsc::Consumer<T>` | exclusive | `write` cursor (its only waiter) | `spsc.rs` |
| `broadcast::Consumer<T>` | exclusive | `wake_seq` (bumped, wakes every waiter) | `broadcast.rs` |
| `mpmc::Consumer<T>` | **shared** | `producer_publication` (wakes as many waiters as items) | `mpmc.rs` |

A public, sealed marker trait, `select::Receive`, documents the consumers `recv` accepts. The hidden traits do the
work:

```rust
/// Operated on only by its owner, through `&mut`.
pub trait ExclusiveEndpoint: Sealed {
    type Output;
    fn try_take(&mut self) -> Option<Self::Output>;   // writes no shared state when empty
    unsafe fn split_wait_word(&mut self) -> (&mut Self, WaitWord<'_>);
}

/// Competes with the other endpoints of its queue, through `&`.
pub trait SharedEndpoint: Sealed {
    type Output;
    fn try_take(&self) -> Option<Self::Output>;       // writes no shared state when empty
    fn wait_word(&self) -> WaitWord<'_>;
}
```

`split_wait_word` hands back the endpoint together with a word that borrows the queue header. The header lives in
the mapped region, not in the endpoint, so the two borrows do not alias. It is `unsafe` because replacing the
endpoint through the returned reference could unmap the header while the word is still in use, and the select
never does that.

`WaitWord` names the futex word and the queue's `Waiters`, plus two extra pieces:
- a **reservation cursor**, for words whose publish wakes only as many waiters as items. These words need the
  hand-off described below.
- a **check cost**. A broadcast check scans every lane, so the spin budget is scaled down by this cost, as
  `broadcast`'s own waits already do.

## Blocking and wake protocol

### Background: the single-word protocol (`futex.rs`)

The two sides follow a Dekker-style ordering:
- **Publisher:** stores the cursor (Release), issues `fence(SeqCst)`, then loads the waiter count. It makes the
  wake syscall only if the count is nonzero.
- **Waiter:** increments the count, issues `fence(SeqCst)`, snapshots the word, rechecks for data, and sleeps with
  `FUTEX_WAIT` on the word's low 32 bits.

Because SeqCst fences are totally ordered, either the publisher sees the waiter, or the waiter's recheck sees the
publication.

### Steps

Every arm has exactly one futex word, fixed for the whole select, so a blocked select waits on the arms' words,
deduplicated by address (at most 32).

| | Step |
|---|---|
| A | **Attempt pass.** Try every arm once. A failed attempt writes no shared queue state. |
| B | **Spin.** Repeat A up to `SPIN_ATTEMPTS / Σ check cost` times; 0 under loom. |
| C | **Register.** Register with every word's `Waiters`, issue one `fence(SeqCst)`, snapshot every word, then recheck with A. |
| D | **Sleep.** If the deadline passed, time out. If any word already moved, skip the syscall. With 1 word, use `FUTEX_WAIT`. With more, use `futex_waitv`: `FUTEX2_SIZE_U32`, no `FUTEX2_PRIVATE` (the words are in shared memory), and an absolute `CLOCK_MONOTONIC` deadline. |
| E | **Wake.** Unregister, issue `fence(SeqCst)`, and go back to A, then C. After a timeout, run A one last time before giving up. |
| F | **Finish.** If the select slept, then for every shared word other than the one the firing arm took from: load `reservation` (Acquire), then `publication`; if they differ, re-issue `wake(word, 1)`. Giving up after a timeout does the same for every shared word. |

**Fallback.** On Linux without `futex_waitv` (before 5.16, or under a seccomp filter that refuses it with `ENOSYS`
or `EPERM`), a select over several words sleeps with `FUTEX_WAIT` on the first word alone, for at most 1 ms at a
time, then makes another pass. A publish on another queue is therefore seen within 1 ms rather than at once, and an
idle select costs no CPU. On `ENOMEM` it waits this way for that one call. Off Linux, a select polls the words with
`yield_now`, as the single-queue waits do there. Either way it registers with every word: a broadcast publish bumps
`wake_seq` only while a waiter is registered, so an unregistered poller would never see the word move.

### Absorbed wakes

An mpmc publish of `n` items wakes at most `n` waiters. A thread in `futex_waitv` is queued on every one of its
words, and the kernel unqueues it from the others only once it runs. So two publishes on different queues can both
pick the same select, which then takes only one item. The other publish's wake is spent, and a plain consumer of
that queue may stay asleep beside the item. Step F passes such wakes on.

Exclusive words need no hand-off. An spsc queue has one consumer, so its wake reaches the only possible waiter. A
broadcast publish wakes every registered waiter.

### Correctness sketch

1. **No lost wake.** Take the final recheck before a sleep. An arm can become ready only through a store to its
   word, and the select registered with every word before that recheck. The per-word Dekker argument then applies
   to each word.
2. **An absorbed wake never strands an item.** A select can owe a wake on word `w` only if a `FUTEX_WAKE` marked it
   while it slept. After E's fence, the next pass sees that publication, and every arm of `w` takes from `w`
   alone.
   - If an arm fires, F re-issues the wake for every other shared word that still holds items. F loads the
     reservation with Acquire: a consumer's claim then carries the publications that consumer saw, so a later item
     is not mistaken for the claimed one.
   - If nothing fires, `w` held no item: someone else claimed it, so nothing is owed. The flag that a sleep
     happened is kept until the next sleep or finish, so a stale read in that pass only delays the hand-off.
3. **No livelock.** Hand-offs happen only when an arm fires or the select gives up, and each one wakes at most one
   waiter per word.
4. **Timeouts hold.** D checks the deadline before anything else, so words that keep moving (items published and
   taken by faster consumers) cannot hold off a timeout.

These interleavings were worked through: a publish racing registration, the recheck, or the queueing of a later
word in `futex_waitv` (`EAGAIN`); two wakes absorbed and one used; a timeout; and the same queue in two arms.

## Shared-memory ABI

Unchanged. The select uses each queue's existing waiter count and futex word.

The mpmc and spsc headers are now initialized by writing the whole struct, as broadcast's already is. The layout
is unchanged. The write constructs loom atomics properly, so the loom models can run over all three queues.

## Verification

- **Unit tests** (`select::tests`):
  - every queue kind, the ready-arm-only rule, `default` and `default(timeout)`;
  - operand forms and arm syntax, control flow in bodies, and 32 arms.
  - Blocking tests fail if the select completes only through its final pass after the watchdog timeout. Without
    that check, a lost wake shows up only as a delay of the full timeout.
- **Stress test** (`competing_selects_and_reads_lose_no_wakes`): three selects over two mpmc queues and a plain
  blocking reader compete. Every item is consumed exactly once, under a watchdog. Looped 300 times in release, with
  the blocking tests, with no failure.
- **Loom models** (`select::loom_tests`, `broadcast::loom_tests::publication_wakes_select`), over real queues:
  - The futex model compares and queues one word at a time, as `futex_waitv` does, so absorption during queueing
    is explored.
  - It reads wakes from the queue rather than from park tokens. Loom discards an unpark that reaches a thread
    blocked on a mutex, so relying on tokens would invent lost wakes.
- **Mutation check of the loom models.** Each mutation below is caught:

  | Mutation | Caught by |
  |---|---|
  | No hand-off | `absorbed_wake_is_handed_off` |
  | No registration fence | every model |
  | spsc on the wrong word | `publication_on_exclusive_queue_wakes_select` |
  | broadcast on the wrong word | `broadcast::loom_tests::publication_wakes_select` |

  The hand-off schedule needs a preemption bound of 3; at 2 the missing hand-off goes unnoticed.
- **Fallback.** The tests also pass with `futex_waitv` forced off. With the fallback that skipped registration,
  `blocks_until_any_kind_of_queue_publishes` fails. With `ENOSYS` injected (`strace --seccomp-bpf -e
  inject=futex_waitv:error=ENOSYS`), an idle two-queue select uses no CPU over a 2 s timeout (busy polling used
  2 s), and a publish on its second queue is seen within about 1 ms.
- **Loom's limits.** The futex model serializes every operation on one mutex, which adds ordering the kernel does
  not provide, and it models neither timeouts nor `EINTR`. `random_index` is 0 under loom, so `select!` is modeled
  as `select_biased!`.
- **Miri** over the non-blocking `select`, `spsc` and `mpmc` tests.

## Extending

A new kind of endpoint needs three things:
1. Implement `ExclusiveEndpoint` or `SharedEndpoint`, and `Sealed`.
2. Implement `Receive` (sealed).
3. Provide a `WaitWord` that its queue reliably wakes, and that it wakes only once the data a `try_take` reads
   is published.

Which constructor to use:
- A word whose wakes can be absorbed (count-limited wakes to competing waiters) must be built with
  `WaitWord::competing`, so that F hands absorbed wakes off.
- Any other word uses `WaitWord::new`.

## Alternatives considered

- **Polling instead of `futex_waitv`.** Kept only as the fallback.
- **A poller that skips registration.** Rejected: broadcast publishes do not move their word for unregistered
  waiters, so such a select never woke on a broadcast publish.
- **Busy polling on Linux without `futex_waitv`.** Rejected: it kept a core busy for as long as the select waited.
- **Coupling select to mpmc.** This was the first cut. The endpoint traits remove it: `select.rs` names no queue
  type.

## Future work

- **Step 2, `send`.** Producers gain a way to wait for room. The mpmc header gains producer waiters on
  `consumer_release`, and every consumer release pays a `fence(SeqCst)` and a load, as every publish already
  does. That cost needs a pinned benchmark first. Room is contended between mpmc producers, so their room wakes
  need the same hand-off as consumers.
- **Step 3, `reserve && recv`.** An `mpmc::ExclusiveProducer` whose reservations write no shared state, so a
  dropped `WritePermit` cancels for free. Arms join exclusive operations with at most one shared operation, last.
  Each arm sleeps on its first operation that is not ready, which can change between passes, so registration is
  re-planned when it does.
- **Broadcast `SliceConsumer`.** It could become a `recv` whose output borrows the consumer.
