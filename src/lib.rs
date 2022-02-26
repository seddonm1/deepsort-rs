#[macro_use]
extern crate lazy_static;

mod bounding_box;
mod detection;
mod iou_matching;
mod kalman_filter;
mod linear_assignment;
mod nn_matching;
mod track;
mod tracker;

pub use bounding_box::BoundingBox;
pub use detection::Detection;
pub use kalman_filter::KalmanFilter;
pub use linear_assignment::{DistanceMetricFn, Match};
pub use nn_matching::{Metric, NearestNeighborDistanceMetric};
pub use track::{MatchSource, Track};
pub use tracker::Tracker;
