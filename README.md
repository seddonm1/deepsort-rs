# deepsort-rs

This is a reimplementation of [Deep SORT](https://github.com/nwojke/deep_sort) (Simple Online and Realtime Tracking with a Deep Association Metric) in pure Rust. The data structures, logic and even comments have been ported from the original source and tests have been created to validate equivalence.

An example of how to use can be found in the benches directory.

Note: this implementation depends on a BLAS library (like `openblas`) which may make this library only work with `linux` targets.