//! Models of `select!` sleeping on several queues at once, over real queues
//! and the futex model, which queues a multi-word wait one word at a time as
//! the kernel does.

use crate::{mpmc, spsc};
use loom::thread;

/// Bounds the larger models, which are too big to explore exhaustively. The
/// stranding schedule `absorbed_wake_is_handed_off` targets needs three
/// preemptions; with two, removing the hand-off goes unnoticed.
const PREEMPTION_BOUND: usize = 3;

fn bounded(model: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound.get_or_insert(PREEMPTION_BOUND);
    builder.check(model);
}

// Tests registration across queues. A publish to either queue racing with
// registration, the recheck, or the multi-word sleep must be observed before
// sleeping or wake the select.
#[test]
fn publication_on_any_queue_wakes_select() {
    loom::model(|| {
        let (_tx_a, rx_a) = mpmc::pair::<usize>(1).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<usize>(1).unwrap();
        let select = thread::spawn(move || {
            crate::select! {
                recv(rx_a) -> value => value,
                recv(rx_b) -> value => value,
            }
        });

        tx_b.try_write(1).unwrap();
        assert_eq!(select.join().unwrap(), 1);
    });
}

// Tests an exclusive endpoint's word. A publish to an spsc queue must wake a
// select that also sleeps on an mpmc queue.
#[test]
fn publication_on_exclusive_queue_wakes_select() {
    loom::model(|| {
        let (mut spsc_tx, mut spsc_rx) = spsc::pair::<usize>(1).unwrap();
        let (_mpmc_tx, mpmc_rx) = mpmc::pair::<usize>(1).unwrap();
        let select = thread::spawn(move || {
            crate::select! {
                recv(spsc_rx) -> value => value,
                recv(mpmc_rx) -> value => value,
            }
        });

        spsc_tx.try_write(1).unwrap();
        assert_eq!(select.join().unwrap(), 1);
    });
}

// Tests the wake hand-off. A select asleep on both queues can be picked by the
// wakes for both items before it runs, yet takes only one. It must pass the
// other wake on, or the second select sleeps beside an unclaimed item.
#[test]
fn absorbed_wake_is_handed_off() {
    bounded(|| {
        let (tx_a, rx_a) = mpmc::pair::<usize>(1).unwrap();
        let (tx_b, rx_b) = mpmc::pair::<usize>(1).unwrap();
        let selects: Vec<_> = (0..2)
            .map(|_| {
                let (rx_a, rx_b) = (rx_a.clone(), rx_b.clone());
                thread::spawn(move || {
                    crate::select_biased! {
                        recv(rx_a) -> value => value,
                        recv(rx_b) -> value => value,
                    }
                })
            })
            .collect();

        tx_a.try_write(1).unwrap();
        tx_b.try_write(2).unwrap();
        let sum: usize = selects
            .into_iter()
            .map(|select| select.join().unwrap())
            .sum();
        assert_eq!(sum, 3);
    });
}

// Tests one queue in two arms. The select must register with the queue once
// and wake on its publish; either arm may then take the item.
#[test]
fn same_queue_in_two_arms() {
    loom::model(|| {
        let (tx, rx) = mpmc::pair::<usize>(1).unwrap();
        let (_tx_b, rx_b) = mpmc::pair::<usize>(1).unwrap();
        let select = thread::spawn(move || {
            crate::select_biased! {
                recv(&rx) -> value => value,
                recv(rx_b) -> value => value,
                recv(&rx) -> value => value,
            }
        });

        tx.try_write(1).unwrap();
        assert_eq!(select.join().unwrap(), 1);
    });
}
