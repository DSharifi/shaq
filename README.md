# shaq

[![Rust CI](https://github.com/anza-xyz/shaq/actions/workflows/ci.yml/badge.svg)](https://github.com/anza-xyz/shaq/actions/workflows/ci.yml)

shaq is a **SHAred Queue**: a simple shared-memory SPSC (Single Producer Single Consumer) and MPMC (Multi Producer Multi Consumer) FIFO queue.
It is designed for efficient inter-thread or inter-process communication using lock-free queues.

`shaq` now supports two backing modes:

- File-backed shared memory via `create` / `join`, for inter-process communication.
- In-process heap-backed queues via `spsc::pair` and `mpmc::pair`, for channel-style usage without file backing.

`shaq::select!` waits on several queues at once and takes an item from one that is ready (trying arms from a
pseudo-random one onwards; `select_biased!` tries them in order): `recv` works on `spsc`, `mpmc`, and `broadcast`
consumers, in any mix. See [`docs/design/select.md`](docs/design/select.md).
