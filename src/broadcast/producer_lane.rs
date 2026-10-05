//! One producer's lane: a self-contained shared-memory block holding the lane's
//! ownership state, its reserve/publication cursors, the per-consumer reserve
//! limits, and the ring of payloads. See [`ProducerLane`].

use core::alloc::Layout;
use core::mem::{align_of, size_of};
use core::num::NonZeroUsize;
use core::ptr::NonNull;
use core::sync::atomic::{fence, AtomicU64, AtomicUsize, Ordering};
use std::marker::PhantomData;

use crate::broadcast::{InitializedLane, LaneIndex, ProducerId, UnverifiedLane};
use crate::CacheAlignedAtomicSize;

use super::consumer_state::LaneConsumerState;

const LANE_FREE: u64 = 0;
const LANE_CLAIMING: u64 = 1;
const LANE_ACTIVE: u64 = 2;
const LANE_RELEASED: u64 = 3;

/// Fixed-size head of a producer-lane block.
#[repr(C)]
pub(super) struct LaneHeader {
    /// Lane ownership: `LANE_FREE`, `LANE_CLAIMING`, `LANE_ACTIVE`, or `LANE_RELEASED`.
    state: AtomicU64,
    /// [`ProducerId`] of the current or most recent owner, meaningful once the
    /// lane is active or released.
    producer_id: AtomicU64,
    /// First sequence of the current or most recent owner's tenure. Sequences
    /// below it were published by an earlier owner whose id is no longer stored.
    tenure_start: AtomicUsize,
    /// Count of messages refused by backpressure over the lane's lifetime. On
    /// its own cache line: the producer bumps it on every refused reserve, which
    /// would otherwise evict the ownership fields consumers read.
    rejected_items: CacheAlignedAtomicU64,
    /// Claimed-up-to sequence: advanced before a ring cell is written.
    producer_reservation: CacheAlignedAtomicSize,
    /// Visible-up-to sequence: advanced after a ring cell is written; consumers
    /// read sequences `< producer_publication`.
    producer_publication: CacheAlignedAtomicSize,
}

/// `AtomicU64` with 64-byte alignment.
#[derive(Default)]
#[repr(C, align(64))]
struct CacheAlignedAtomicU64 {
    inner: AtomicU64,
}

impl core::ops::Deref for CacheAlignedAtomicU64 {
    type Target = AtomicU64;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A single producer's lane.
///
/// The lane holds ownership state, the reserve/publication cursors, the
/// per-lane consumer reserve-limit state, and the ring of fixed-size payload
/// cells. The owning
/// producer mutates the ring and producer cursors (`&mut self`); consumers read
/// published payloads and publish their own progress through [`LaneConsumerState`].
///
/// Block layout: `LaneHeader`, then `[CacheAlignedAtomicSize; consumer_slots]`
/// limits (one per cache line), then the payload ring.
pub(crate) struct ProducerLane {
    header: NonNull<LaneHeader>,
    consumer_state: LaneConsumerState,
    ring: NonNull<u8>,
    payload_size: usize,

    mask: usize, // capacity - 1
}

/// A borrowed view of one producer lane's metadata.
///
/// The view cannot outlive the broadcast mapping from which it was obtained.
#[derive(Clone, Copy)]
pub struct LaneMetadata<'a> {
    header: &'a LaneHeader,
    lane: LaneIndex<InitializedLane>,
    /// First sequence of the message(s) this view describes, or `None` for a
    /// lane-level view.
    sequence: Option<usize>,
}

impl<'a> LaneMetadata<'a> {
    /// Builds a metadata view if the lane has completed acquisition.
    pub(super) fn try_new(header: &'a LaneHeader, lane: LaneIndex<UnverifiedLane>) -> Option<Self> {
        let state = header.state.load(Ordering::Acquire);

        let lane_is_acquired = matches!(state, LANE_ACTIVE | LANE_RELEASED);
        if !lane_is_acquired {
            return None;
        }

        Some(Self {
            header,
            lane: LaneIndex {
                index: lane.get(),
                _state: PhantomData,
            },
            sequence: None,
        })
    }

    /// Builds a metadata view for the published message(s) starting at
    /// `sequence`.
    pub(super) fn for_message(
        header: &'a LaneHeader,
        lane: LaneIndex<InitializedLane>,
        sequence: usize,
    ) -> Self {
        Self {
            header,
            lane,
            sequence: Some(sequence),
        }
    }

    /// The index of this producer lane.
    #[inline]
    pub fn lane(&self) -> usize {
        self.lane.get()
    }

    /// The producer chosen [`ProducerId`] for this view.
    ///
    /// For a view borrowed from a read guard or batch, this is the id of the
    /// producer that published the value(s), or `None` if the lane has since
    /// been handed to another producer and that id is no longer stored. For a
    /// lane-level view, this is the lane's current or most recent owner.
    #[inline]
    pub fn producer_id(&self) -> Option<ProducerId> {
        // Pairs with the release store in `ProducerLane::try_acquire`: observing
        // a tenure's id also observes that tenure's start or a later one, so a
        // value is never attributed to a producer that took over after it.
        let producer_id = ProducerId::new(self.header.producer_id.load(Ordering::Acquire));
        let Some(sequence) = self.sequence else {
            return Some(producer_id);
        };
        let tenure_start = self.header.tenure_start.load(Ordering::Relaxed);
        (sequence >= tenure_start).then_some(producer_id)
    }

    /// Count of messages refused by backpressure over this lane's lifetime.
    #[inline]
    pub fn rejected_items(&self) -> u64 {
        self.header.rejected_items.load(Ordering::Relaxed)
    }
}

impl core::fmt::Debug for LaneMetadata<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaneMetadata")
            .field("lane", &self.lane())
            .field("producer_id", &self.producer_id())
            .field("rejected_items", &self.rejected_items())
            .finish()
    }
}

#[inline]
const fn consumer_state_offset() -> usize {
    size_of::<LaneHeader>().next_multiple_of(LaneConsumerState::block_align())
}

pub(crate) fn ring_offset_for_payload(
    consumer_slots: usize,
    payload_layout: Layout,
) -> Option<usize> {
    if payload_layout.align() > ProducerLane::block_align() {
        return None;
    }

    consumer_state_offset()
        .checked_add(LaneConsumerState::block_size(consumer_slots)?)?
        .checked_next_multiple_of(payload_layout.align())
}

pub(crate) fn block_size_for_payload(
    capacity: u32,
    consumer_slots: usize,
    payload_layout: Layout,
) -> Option<usize> {
    ring_offset_for_payload(consumer_slots, payload_layout)?
        .checked_add((capacity as usize).checked_mul(payload_layout.size())?)
}

impl ProducerLane {
    pub(crate) const fn block_align() -> usize {
        align_of::<LaneHeader>()
    }

    /// Initializes a lane block: ownership/cursors zeroed, every consumer slot
    /// free. The ring is left uninitialized (each cell is written before it is
    /// published, and read only after).
    ///
    /// # Safety
    /// - `block` must point at a [`block_size_for_payload`] region for the
    ///   queue's `(capacity, consumer_slots, payload_layout)` and be initialized
    ///   at most once.
    pub(crate) unsafe fn init(block: NonNull<u8>, consumer_slots: usize) {
        let header = LaneHeader {
            state: AtomicU64::new(LANE_FREE),
            producer_id: AtomicU64::new(0),
            tenure_start: AtomicUsize::new(0),
            rejected_items: CacheAlignedAtomicU64::default(),
            producer_reservation: CacheAlignedAtomicSize::default(),
            producer_publication: CacheAlignedAtomicSize::default(),
        };
        // SAFETY: `block` begins with a `LaneHeader`.
        unsafe { block.cast().write(header) };
        // SAFETY: layout reserves `consumer_slots` aligned slots here.
        let consumer_state = unsafe { block.byte_add(consumer_state_offset()) };
        // SAFETY: freshly initialized lane block; consumer slots are initialized once.
        unsafe { LaneConsumerState::init(consumer_state, consumer_slots) };
    }

    /// Builds a lane view over an initialized block.
    ///
    /// # Safety
    /// - `block` must reference a block initialized by [`Self::init`] with the
    ///   same `consumer_slots`, sized for `(capacity, payload_layout)`, alive
    ///   for the view's use.
    pub(crate) unsafe fn from_block(
        block: NonNull<u8>,
        capacity: u32,
        consumer_slots: usize,
        payload_layout: Layout,
    ) -> Self {
        debug_assert!(capacity.is_power_of_two());

        let header = block.cast();
        // SAFETY: `block` is an initialized region - it must be large enough
        //         for consumer state to fit (if init succeeded).
        let consumer_state_block = unsafe { block.byte_add(consumer_state_offset()) };
        // SAFETY: `block` is an initialized region - it must be large enough
        //         for consumer state to fit (if init succeeded).
        let consumer_state = unsafe {
            LaneConsumerState::from_block(consumer_state_block, consumer_slots, capacity as usize)
        };
        let ring_offset =
            ring_offset_for_payload(consumer_slots, payload_layout).expect("validated_lane layout");
        // SAFETY: `block` is an initialized region - it must be large enough
        //         for ring data to fit (if init succeeded).
        let ring = unsafe { block.byte_add(ring_offset) };

        Self {
            header,
            consumer_state,
            ring,
            payload_size: payload_layout.size(),
            mask: (capacity as usize).wrapping_sub(1),
        }
    }

    /// Returns borrowed metadata for the published message(s) starting at
    /// `sequence` on this lane.
    #[inline]
    pub(crate) fn message_metadata(
        &self,
        lane: LaneIndex<InitializedLane>,
        sequence: usize,
    ) -> LaneMetadata<'_> {
        LaneMetadata::for_message(self.header(), lane, sequence)
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.mask.wrapping_add(1)
    }

    #[inline]
    pub(crate) fn mask(&self, sequence: usize) -> usize {
        sequence & self.mask
    }

    #[inline]
    fn header(&self) -> &LaneHeader {
        // SAFETY: The lane has an initialized header that lives for the lane's lifetime.
        unsafe { self.header.as_ref() }
    }

    #[inline]
    pub(crate) fn consumer_state(&self) -> LaneConsumerState {
        self.consumer_state
    }

    /// Pointer to the ring cell holding `sequence` — used by the producer to
    /// write a reserved cell and by consumers to read a published one.
    #[inline]
    pub(crate) fn payload_ptr(&self, sequence: usize) -> NonNull<u8> {
        let offset = self.mask(sequence).wrapping_mul(self.payload_size);
        // SAFETY: `sequence & mask < capacity`; the ring has `capacity` cells of
        // `payload_size` bytes. Zero-sized payloads always point at the ring base.
        unsafe { self.ring.byte_add(offset) }
    }

    /// Claims a free or released lane for a producer, starting a new tenure
    /// that carries `producer_id`. Returns `false` if already owned.
    #[must_use]
    pub(crate) fn try_acquire(&self, producer_id: ProducerId) -> bool {
        let acquire_result =
            self.header()
                .state
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                    matches!(state, LANE_FREE | LANE_RELEASED).then_some(LANE_CLAIMING)
                });

        if acquire_result.is_err() {
            return false;
        }

        // A lane is only released with no unpublished reservation, so this
        // tenure's first value lands at the reservation frontier.
        self.header()
            .tenure_start
            .store(self.reserved(), Ordering::Relaxed);
        // Store the id after the tenure start; see `LaneMetadata::producer_id`.
        self.header()
            .producer_id
            .store(producer_id.get(), Ordering::Release);

        self.header().state.store(LANE_ACTIVE, Ordering::Release);

        true
    }

    /// Releases the lane for reuse, preserving its cursors, rejection count,
    /// and the last owner's id.
    pub(crate) fn release(&self) {
        // A forgotten write guard can leave uninitialized cells reserved. Reuse
        // would publish that gap, while rewinding the reservation cursor would
        // break the concurrent consumer-join handshake. Keep this lane claimed.
        if self.reserved() != self.published() {
            return;
        }
        let _ = self.header().state.compare_exchange(
            LANE_ACTIVE,
            LANE_RELEASED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// Reserves `count` consecutive sequences for writing, returning the first.
    /// `None` on backpressure: the batch would overwrite a cell an active
    /// consumer has not yet read, or it exceeds the ring capacity.
    /// On success this returns Some(seqnum) - with seqnum being
    /// the starting sequence number of the reservation.
    ///
    /// Write each reserved cell via [`Self::payload_ptr`], then [`Self::publish`].
    pub(crate) fn try_reserve(&mut self, count: NonZeroUsize) -> Option<usize> {
        if count.get() > self.capacity() {
            return None;
        }
        let start = self.header().producer_reservation.load(Ordering::Acquire);
        // Producer half of the join handshake: order the previous reserve's
        // `producer_reservation` store before this reserve's limit loads. A
        // racing consumer publishes a limit from its first reservation sample,
        // fences, then samples again. Either this reserve observes that limit or
        // the consumer observes our committed frontier. If both race with this
        // reserve's later store, the consumer starts at `start`, making this
        // reservation future data rather than an overwrite.
        fence(Ordering::SeqCst);
        // Each slot already stores `next_to_read + capacity`, so the gate is a
        // plain comparison: rejecting once the batch would reach a sequence a
        // consumer still needs. Unowned slots sit at the top, so they never
        // gate.
        if start.wrapping_add(count.get()) > self.consumer_state.reserve_limit() {
            self.header()
                .rejected_items
                .fetch_add(count.get() as u64, Ordering::Relaxed);
            return None;
        }
        // Claim before the writes; consumers only read `< producer_publication`.
        self.header()
            .producer_reservation
            .store(start.wrapping_add(count.get()), Ordering::Release);
        Some(start)
    }

    /// Publishes `start..start + count`, making it visible to consumers. Call
    /// after the cells are written.
    pub(crate) fn publish(&mut self, start: usize, count: NonZeroUsize) {
        self.header()
            .producer_publication
            .store(start.wrapping_add(count.get()), Ordering::Release);
    }
    #[inline]
    pub(crate) fn published(&self) -> usize {
        self.header().producer_publication.load(Ordering::Acquire)
    }

    #[inline]
    pub(crate) fn reserved(&self) -> usize {
        self.header().producer_reservation.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shmem::Region;
    use std::num::NonZeroUsize;

    type Payload = u64;

    const BOGUS_PRODUCER_ID: ProducerId = ProducerId::new(1111);

    /// Allocates and initializes a standalone lane block.
    fn lane(capacity: u32, consumer_slots: usize) -> (std::sync::Arc<Region>, ProducerLane) {
        let payload_layout = Layout::new::<Payload>();
        let size = block_size_for_payload(capacity, consumer_slots, payload_layout).unwrap();
        let region = Region::alloc(NonZeroUsize::new(size).unwrap()).unwrap();
        // SAFETY: freshly allocated block, initialized once.
        unsafe { ProducerLane::init(region.addr(), consumer_slots) };
        // SAFETY: just initialized with these parameters.
        let lane = unsafe {
            ProducerLane::from_block(region.addr(), capacity, consumer_slots, payload_layout)
        };
        (region, lane)
    }

    fn read(lane: &ProducerLane, sequence: usize) -> Payload {
        // SAFETY: `sequence` was published, so its cell is initialized.
        unsafe { lane.payload_ptr(sequence).cast().read() }
    }

    fn join_consumer(lane: &ProducerLane, consumer_index: usize) -> usize {
        let consumer_state = lane.consumer_state();
        consumer_state.join(consumer_index, || lane.reserved())
    }

    fn metadata(lane: &ProducerLane) -> LaneMetadata<'_> {
        LaneMetadata::try_new(lane.header(), LaneIndex::new(0)).expect("lane is active or released")
    }

    /// The producer id reported for the published message at `sequence`.
    fn message_producer_id(lane: &ProducerLane, sequence: usize) -> Option<ProducerId> {
        let lane_index = LaneIndex {
            index: 0,
            _state: PhantomData,
        };
        lane.message_metadata(lane_index, sequence).producer_id()
    }

    /// Reserves, writes, and publishes one value; `false` on backpressure.
    fn publish_value(lane: &mut ProducerLane, value: Payload) -> bool {
        let one = NonZeroUsize::new(1).unwrap();
        let Some(start) = lane.try_reserve(one) else {
            return false;
        };
        // SAFETY: the cell is reserved and not yet published.
        unsafe { lane.payload_ptr(start).cast().write(value) };
        lane.publish(start, one);
        true
    }

    #[test]
    fn lane_ownership_is_exclusive() {
        let (_region, lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        assert!(!lane.try_acquire(BOGUS_PRODUCER_ID));
    }

    #[test]
    fn released_lane_can_be_reclaimed() {
        let (_region, lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        lane.release();
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        assert!(!lane.try_acquire(BOGUS_PRODUCER_ID));
    }

    #[test]
    fn reacquiring_starts_a_tenure_at_the_reservation_frontier() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        for value in 0..3u64 {
            assert!(publish_value(&mut lane, value));
        }
        lane.release();

        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));

        assert_eq!(lane.header().tenure_start.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn message_metadata_reports_only_the_publishing_tenure() {
        let (_region, mut lane) = lane(8, 1);
        let first = ProducerId::new(41);
        let second = ProducerId::new(42);
        assert!(lane.try_acquire(first));
        assert!(publish_value(&mut lane, 0));
        assert!(publish_value(&mut lane, 1));
        assert_eq!(message_producer_id(&lane, 1), Some(first));
        lane.release();

        assert!(lane.try_acquire(second));
        assert!(publish_value(&mut lane, 2));

        assert_eq!(message_producer_id(&lane, 0), None);
        assert_eq!(message_producer_id(&lane, 1), None);
        assert_eq!(message_producer_id(&lane, 2), Some(second));
    }

    #[test]
    fn empty_tenure_does_not_claim_earlier_messages() {
        let (_region, mut lane) = lane(8, 1);
        let last = ProducerId::new(43);
        assert!(lane.try_acquire(ProducerId::new(41)));
        assert!(publish_value(&mut lane, 0));
        lane.release();
        assert!(lane.try_acquire(ProducerId::new(42)));
        lane.release();

        assert!(lane.try_acquire(last));
        assert!(publish_value(&mut lane, 1));

        assert_eq!(message_producer_id(&lane, 0), None);
        assert_eq!(message_producer_id(&lane, 1), Some(last));
    }

    #[test]
    fn never_owned_lane_has_not_completed_acquisition() {
        let (_region, lane) = lane(4, 1);

        let metadata = LaneMetadata::try_new(lane.header(), LaneIndex::new(0));

        assert!(metadata.is_none());
    }

    #[test]
    fn claiming_lane_has_not_completed_acquisition() {
        let (_region, lane) = lane(4, 1);
        let header = lane.header();
        header
            .state
            .compare_exchange(
                LANE_FREE,
                LANE_CLAIMING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .expect("lane is free");
        header.producer_id.store(42, Ordering::Relaxed);

        let metadata = LaneMetadata::try_new(lane.header(), LaneIndex::new(0));

        assert!(metadata.is_none());
    }

    #[test]
    fn acquiring_installs_the_producer_id() {
        let (_region, lane) = lane(4, 1);
        let producer_id = ProducerId::new(42);

        let _ = lane.try_acquire(producer_id);

        assert_eq!(metadata(&lane).producer_id(), Some(producer_id));
    }

    #[test]
    fn refused_reserves_count_rejected_items() {
        let (_region, mut lane) = lane(4, 1);
        let _ = lane.try_acquire(BOGUS_PRODUCER_ID);
        let _ = join_consumer(&lane, 0);
        for value in 0..4u64 {
            let _ = publish_value(&mut lane, value);
        }

        let _ = publish_value(&mut lane, 99);
        let _ = lane.try_reserve(NonZeroUsize::new(2).unwrap());

        assert_eq!(metadata(&lane).rejected_items(), 3);
    }

    #[test]
    fn released_lane_keeps_the_last_owner_id() {
        let (_region, lane) = lane(4, 1);
        let producer_id = ProducerId::new(42);
        let _ = lane.try_acquire(producer_id);

        lane.release();

        assert_eq!(metadata(&lane).producer_id(), Some(producer_id));
    }

    #[test]
    fn released_lane_keeps_the_rejected_items_count() {
        let (_region, mut lane) = lane(4, 1);
        let _ = lane.try_acquire(ProducerId::new(42));
        let _ = join_consumer(&lane, 0);
        for value in 0..4u64 {
            let _ = publish_value(&mut lane, value);
        }
        let _ = publish_value(&mut lane, 99);

        lane.release();

        assert_eq!(metadata(&lane).rejected_items(), 1);
    }

    #[test]
    fn publishes_and_advances_cursors() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        for value in 0..4u64 {
            assert!(publish_value(&mut lane, value * 10));
        }
        assert_eq!(lane.published(), 4);
        assert_eq!(lane.reserved(), 4);
        for seq in 0..4usize {
            assert_eq!(read(&lane, seq), seq as u64 * 10);
        }
    }

    #[test]
    fn reserves_and_publishes_a_batch() {
        let (_region, mut lane) = lane(8, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        let count = NonZeroUsize::new(3).unwrap();
        let start = lane.try_reserve(count).expect("reserve");
        for offset in 0..count.get() {
            // SAFETY: each cell in the batch is reserved and unpublished.
            unsafe {
                lane.payload_ptr(start.wrapping_add(offset))
                    .cast()
                    .write((offset as u64) + 1)
            };
        }
        // Reserved but not yet visible.
        assert_eq!(lane.reserved(), 3);
        assert_eq!(lane.published(), 0);
        lane.publish(start, count);
        assert_eq!(lane.published(), 3);
        for offset in 0..3usize {
            assert_eq!(read(&lane, offset), offset as u64 + 1);
        }
    }

    #[test]
    fn reserve_rejects_count_above_capacity() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        assert!(lane.try_reserve(NonZeroUsize::new(5).unwrap()).is_none());
        // A caller error is not backpressure, so it is not counted as
        // rejected items.
        assert_eq!(metadata(&lane).rejected_items(), 0);
    }

    #[test]
    fn no_active_consumers_allows_free_overwrite() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        // Publish well past one revolution; with no active consumer there is
        // nothing to protect, so every reserve succeeds.
        for value in 0..16u64 {
            assert!(publish_value(&mut lane, value));
        }
        assert_eq!(lane.published(), 16);
        // The ring holds the most recent generation (sequences 12..16).
        for seq in 12..16usize {
            assert_eq!(read(&lane, seq), seq as u64);
        }
    }

    #[test]
    fn backpressure_when_consumer_lags() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        // Join consumer 0; nothing published yet, so it starts at sequence 0.
        assert_eq!(join_consumer(&lane, 0), 0);

        // Fill the ring; the consumer has read nothing, so the next reserve laps.
        for value in 0..4u64 {
            assert!(publish_value(&mut lane, value));
        }
        assert!(!publish_value(&mut lane, 99));

        // Consumer consumes sequences 0 and 1; two cells free up.
        lane.consumer_state().set_cursor(0, 2);
        assert!(publish_value(&mut lane, 100));
        assert!(publish_value(&mut lane, 101));
        // Now full again relative to the watermark (cursor 2, capacity 4).
        assert!(!publish_value(&mut lane, 102));

        // The recycled cells hold the new payloads.
        assert_eq!(read(&lane, 4), 100);
        assert_eq!(read(&lane, 5), 101);
        assert_eq!(lane.published(), 6);
    }

    #[test]
    fn released_consumer_no_longer_constrains() {
        let (_region, mut lane) = lane(4, 1);
        assert!(lane.try_acquire(BOGUS_PRODUCER_ID));
        assert_eq!(join_consumer(&lane, 0), 0);
        for value in 0..4u64 {
            assert!(publish_value(&mut lane, value));
        }
        assert!(!publish_value(&mut lane, 99));

        // Releasing the slot removes the constraint.
        lane.consumer_state().release(0);
        assert!(publish_value(&mut lane, 99));
    }
}
