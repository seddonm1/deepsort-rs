use std::collections::HashSet;

use crate::{track::TrackState, *};
use anyhow::{Ok, Result};
use ndarray::*;

/// This is the multi-target tracker.
///
///
/// # Examples
///
/// ```
/// use deepsort_rs::{BoundingBox, Detection, Metric, NearestNeighborDistanceMetric, Tracker};
///
/// // instantiate tracker with default parameters
/// let mut tracker = Tracker::default();
///
/// // create a detection
/// // feature vector from a metric learning model output
/// let detection = Detection::new(
///     None,
///     BoundingBox::new(0.0, 0.0, 5.0, 5.0),
///     1.0,
///     None,
///     None,
///     Some(vec![0.0; 128]),
/// );
///
/// // predict then add 0..n detections
/// &tracker.predict();
/// &tracker.update(&[detection]);
///
/// // print predictions
/// for track in tracker.tracks() {
///     println!(
///         "{} {} {:?} {:?}",
///         track.track_id(),
///         track.detection().confidence(),
///         track.state(),
///         track.bbox().to_tlwh(),
///     );
/// };
///```
#[derive(Debug)]
pub struct Tracker {
    /// Gating threshold for intersection over union. Associations with cost larger than this value are disregarded.
    max_iou_distance: f32,
    /// Maximum number of missed misses before a track is deleted.
    max_age: usize,
    /// A Kalman filter to filter target trajectories in image space.
    kf: KalmanFilter,
    /// The list of active tracks at the current time step.
    tracked_tracks: HashSet<Track>,
    /// The list of active tracks at the current time step.
    lost_tracks: HashSet<Track>,
    /// The list of active tracks at the current time step.
    removed_tracks: HashSet<Track>,
    /// Used to allocate identifiers to new tracks.
    next_id: usize,
    /// Confidence threshold for performing primary or secondary matching.
    track_threshold: f32,
    /// Confidence threshold for creating new detections.
    detection_threshold: f32,
    /// First run allows tracking from first frame.
    initial_run: bool,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

impl Tracker {
    /// Returns a new Tracker
    ///
    /// # Arguments
    ///
    /// * `max_iou_distance`: Gating threshold for intersection over union. Associations with cost larger than this value are disregarded. Default `0.9`.
    /// * `max_age`: Maximum number of missed misses before a track is deleted. Default `30`.
    /// * `track_threshold`: Confidence threshold for performing primary or secondary matching. Default `0.6`.
    pub fn new(
        max_iou_distance: Option<f32>,
        max_age: Option<usize>,
        track_threshold: Option<f32>,
    ) -> Tracker {
        Tracker {
            max_iou_distance: max_iou_distance.unwrap_or(0.9),
            track_threshold: track_threshold.unwrap_or(0.6),
            detection_threshold: track_threshold.unwrap_or(0.6) + 0.1,
            max_age: max_age.unwrap_or(30),
            kf: KalmanFilter::default(),
            tracked_tracks: HashSet::new(),
            lost_tracks: HashSet::new(),
            removed_tracks: HashSet::new(),
            next_id: 1,
            initial_run: true,
        }
    }

    /// Set max_iou_distance
    pub fn with_max_iou_distance(&mut self, max_iou_distance: f32) -> &mut Self {
        self.max_iou_distance = max_iou_distance;
        self
    }

    /// Set max_age
    pub fn with_max_age(&mut self, max_age: usize) -> &mut Self {
        self.max_age = max_age;
        self
    }

    /// Return the tracked_tracks
    pub fn tracked_tracks(&self) -> &HashSet<Track> {
        &self.tracked_tracks
    }

    /// Return the lost_tracks
    pub fn lost_tracks(&self) -> &HashSet<Track> {
        &self.lost_tracks
    }

    /// Return the removed_tracks
    pub fn removed_tracks(&self) -> &HashSet<Track> {
        &self.removed_tracks
    }

    /// Perform measurement update and track management.
    ///
    /// # Parameters
    ///
    /// * `detections`: A list of detections at the current time step.
    pub fn update(&mut self, detections: Vec<Detection>) -> Result<Vec<&Track>> {
        let mut new_tracked_tracks = HashSet::new();
        let mut new_lost_tracks = HashSet::new();
        let mut new_removed_tracks = HashSet::new();

        // Split detections into high and low confidence
        let (high_detections, low_detections): (Vec<Detection>, Vec<Detection>) = detections
            .into_iter()
            .filter(|detection| detection.confidence() > 0.1)
            .partition(|detection| detection.confidence() > self.track_threshold);

        let tracked_tracks = std::mem::take(&mut self.tracked_tracks);
        let lost_tracks = std::mem::take(&mut self.lost_tracks);

        // Split track set into confirmed and unconfirmed tracks.
        let (mut track_pool, unconfirmed_tracks): (Vec<Track>, Vec<Track>) = tracked_tracks
            .into_iter()
            .partition(|track| track.is_activated());

        track_pool.extend(lost_tracks);
        track_pool.dedup();
        track_pool
            .iter_mut()
            .for_each(|track| track.predict(&self.kf));

        // Step 1
        // Associate high confidence detections with confimed tracks using IoU.
        let (high_iou_confirmed_matches, unmatched_tracks, unmatched_high_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::intersection_over_union_cost(),
                self.max_iou_distance,
                track_pool,
                high_detections,
            )?;

        // Update tracks
        high_iou_confirmed_matches
            .into_iter()
            .try_for_each(|iou_match| {
                let Match {
                    mut track,
                    detection,
                    distance,
                } = iou_match;

                if track.is_tracked() {
                    track.update(&self.kf, detection, Some(MatchSource::IoU { distance }))?;
                    new_tracked_tracks.insert(track);
                } else {
                    track.re_activate(&self.kf, detection, Some(MatchSource::IoU { distance }))?;
                    new_tracked_tracks.insert(track);
                }

                Ok(())
            })?;

        // Step 2
        // Associate low confidence detections with unmatched confirmed tracks using IoU.
        let (low_iou_confirmed_matches, unmatched_tracks, _unmatched_low_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::intersection_over_union_cost(),
                0.5,
                unmatched_tracks,
                low_detections,
            )?;

        // Update tracks
        low_iou_confirmed_matches
            .into_iter()
            .try_for_each(|iou_match| {
                let Match {
                    mut track,
                    detection,
                    distance,
                } = iou_match;

                if track.is_tracked() {
                    track.update(&self.kf, detection, Some(MatchSource::IoU { distance }))?;
                    new_tracked_tracks.insert(track);
                } else {
                    track.re_activate(&self.kf, detection, Some(MatchSource::IoU { distance }))?;
                    new_tracked_tracks.insert(track);
                }

                Ok(())
            })?;

        // mark unmatched tracks as lost
        unmatched_tracks.into_iter().for_each(|mut track| {
            if !track.is_lost() {
                track.mark_lost();
            }
            new_lost_tracks.insert(track);
        });

        // Step 3
        // Associate unmatched high confidence detections with unconfirmed tracks using IoU.
        let (iou_unconfirmed_matches, unmatched_unconfirmed_tracks, unmatched_high_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::intersection_over_union_cost(),
                0.7,
                unconfirmed_tracks,
                unmatched_high_detections,
            )?;

        // Update matched tracks with the detection
        iou_unconfirmed_matches
            .into_iter()
            .try_for_each(|iou_match| {
                let Match {
                    mut track,
                    detection,
                    distance,
                } = iou_match;

                track.update(&self.kf, detection, Some(MatchSource::IoU { distance }))?;
                new_tracked_tracks.insert(track);

                Ok(())
            })?;

        // Remove any unmatched tracks
        unmatched_unconfirmed_tracks
            .into_iter()
            .for_each(|mut track| {
                track.mark_removed();
                new_removed_tracks.insert(track);
            });

        // Step 4
        // Initialize new tracks that are above the detection threshold
        unmatched_high_detections.into_iter().for_each(|detection| {
            if detection.confidence() > self.detection_threshold {
                let track = self.activate(detection);
                new_tracked_tracks.insert(track);
            }
        });

        // Step 5
        // Remove any tracks that have been lost for n frames
        let (new_lost_tracks, expired_lost_tracks): (HashSet<Track>, HashSet<Track>) =
            new_lost_tracks
                .into_iter()
                .partition(|track| track.time_since_update() <= self.max_age);

        // Step 6
        // Update the state for next run
        _ = std::mem::replace(&mut self.tracked_tracks, new_tracked_tracks);
        self.lost_tracks.extend(new_lost_tracks);
        self.removed_tracks.extend(new_removed_tracks);
        self.removed_tracks.extend(expired_lost_tracks);

        // cannot be intial run anymore
        self.initial_run = false;

        Ok(self
            .tracked_tracks
            .iter()
            .filter(|track| track.is_activated())
            .collect())
    }

    fn activate(&mut self, detection: Detection) -> Track {
        let (mean, covariance) = self.kf.initiate(detection.bbox());
        let features = detection
            .feature()
            .as_ref()
            .map(|feature| feature.clone().insert_axis(Axis(0)));
        let track = Track::new(
            TrackState::Tracked,
            self.initial_run,
            mean,
            covariance,
            self.next_id,
            detection,
            features,
        );
        self.next_id += 1;
        track
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     use anyhow::{Ok, Result};
//     use ndarray::*;
//     use rand::prelude::*;
//     use rand_distr::Normal;
//     use rand_pcg::{Lcg64Xsh32, Pcg32};

//     /// Returns a psuedo-random (deterministic) f32 between -0.5 and +0.5
//     fn next_f32(rng: &mut Lcg64Xsh32) -> f32 {
//         (rng.next_u32() as f64 / u32::MAX as f64) as f32 - 0.5
//     }

//     /// Returns a vec of length n with a normal distribution
//     fn normal_vec(rng: &mut Lcg64Xsh32, mean: f32, std_dev: f32, n: i32) -> Vec<f32> {
//         let normal = Normal::<f32>::new(mean, std_dev).unwrap();
//         (0..n).map(|_| normal.sample(rng)).collect()
//     }

//     #[test]
//     fn tracker_nearest_neighbor() -> Result<()> {
//         let iterations: i32 = 100;

//         // deterministic generator
//         let mut rng = Pcg32::seed_from_u64(0);

//         // create random movement/scale this is a vectors so it can be easily copied to python for comparison
//         let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<_>>();
//         let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

//         // create the feature vectors
//         let d0_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
//         let d1_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
//         let d2_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
//         let d3_feat = normal_vec(&mut rng, 0.0, 1.0, 128);

//         let mut tracker = Tracker::default();

//         for iteration in 0..iterations {
//             // move up to right
//             let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
//             let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
//             let d0 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d0_x,
//                     d0_y,
//                     10.0 + scale_jitter.pop().unwrap(),
//                     10.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 Some(d0_feat.clone()),
//             );

//             // move down to left
//             let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
//             let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
//             let d1 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d1_x,
//                     d1_y,
//                     8.0 + scale_jitter.pop().unwrap(),
//                     8.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 Some(d1_feat.clone()),
//             );

//             // move up to left
//             let d2_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
//             let d2_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
//             let d2 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d2_x,
//                     d2_y,
//                     6.0 + scale_jitter.pop().unwrap(),
//                     6.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 Some(d2_feat.clone()),
//             );

//             // move up and right
//             let d3_x = 0.0 + (iteration as f32 * 0.1) + movement_jitter.pop().unwrap();
//             let d3_y = 0.0 + ((iteration - 50) as f32) + movement_jitter.pop().unwrap();
//             let d3 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d3_x,
//                     d3_y,
//                     5.0 + scale_jitter.pop().unwrap(),
//                     5.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 Some(d3_feat.clone()),
//             );

//             tracker.predict();

//             // add and remove detections over the sequence
//             match iteration {
//                 _ if iteration >= 60 => {
//                     tracker.update(&[d2, d3])?;
//                 }
//                 _ if iteration >= 30 => {
//                     tracker.update(&[d0, d2])?;
//                 }
//                 _ => {
//                     tracker.update(&[d0, d1])?;
//                 }
//             }

//             // for debugging
//             // for track in &tracker.tracks {
//             //     println!(
//             //         "{}: {:?} {:?} {:?} {:?} {:?}",
//             //         iteration,
//             //         track.track_id(),
//             //         track.state(),
//             //         track.bbox().to_tlwh(),
//             //         tracker
//             //             .metric
//             //             .track_features(*track.track_id())
//             //             .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
//             //             .nrows(),
//             //         track.match_source(),
//             //     );
//             // }
//         }

//         assert_eq!(tracker.tracks.len(), 2);

//         let track = tracker
//             .tracks
//             .iter()
//             .filter(|track| track.track_id() == 3)
//             .collect::<Vec<_>>();
//         let track = track.first().unwrap();
//         assert!(track.is_confirmed());
//         assert_eq!(
//             track.bbox().to_tlwh(),
//             arr1::<f32>(&[99.12867, 1.0377614, 6.1343956, 6.1184144])
//         );

//         let track = tracker
//             .tracks
//             .iter()
//             .filter(|track| track.track_id() == 4)
//             .collect::<Vec<_>>();
//         let track = track.first().unwrap();
//         assert!(track.is_confirmed());
//         assert_eq!(
//             track.bbox().to_tlwh(),
//             arr1::<f32>(&[9.640262, 48.84033, 5.1905212, 5.113961])
//         );

//         Ok(())
//     }

//     #[test]
//     fn tracker_intersection_over_union() -> Result<()> {
//         let iterations: i32 = 100;

//         // deterministic generator
//         let mut rng = Pcg32::seed_from_u64(0);

//         // create random movement/scale this is a vectors so it can be easily copied to python for comparison
//         let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<_>>();
//         let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

//         let mut tracker = Tracker::default();

//         for iteration in 0..iterations {
//             // move up to right
//             let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
//             let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
//             let d0 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d0_x,
//                     d0_y,
//                     10.0 + scale_jitter.pop().unwrap(),
//                     10.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             );

//             // move down to left
//             let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
//             let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
//             let d1 = Detection::new(
//                 None,
//                 BoundingBox::new(
//                     d1_x,
//                     d1_y,
//                     8.0 + scale_jitter.pop().unwrap(),
//                     8.0 + scale_jitter.pop().unwrap(),
//                 ),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             );

//             tracker.predict();
//             tracker.update(&[d0, d1])?;

//             // for debugging
//             // for track in &tracker.tracks {
//             //     println!(
//             //         "{}: {:?} {:?} {:?} {:?} {:?}",
//             //         iteration,
//             //         track.track_id(),
//             //         track.state(),
//             //         track.bbox().to_tlwh(),
//             //         tracker
//             //             .metric
//             //             .track_features(*track.track_id())
//             //             .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
//             //             .nrows(),
//             //         track.match_source(),
//             //     );
//             // }
//         }

//         assert_eq!(tracker.tracks.len(), 2);

//         let track = tracker
//             .tracks
//             .iter()
//             .filter(|track| track.track_id() == 1)
//             .collect::<Vec<_>>();
//         let track = track.first().unwrap();
//         assert!(track.is_confirmed());
//         assert_eq!(
//             track.bbox().to_tlwh(),
//             arr1::<f32>(&[99.26583, 98.69391, 9.864473, 9.7438])
//         );

//         let track = tracker
//             .tracks
//             .iter()
//             .filter(|track| track.track_id() == 2)
//             .collect::<Vec<_>>();
//         let track = track.first().unwrap();
//         assert!(track.is_confirmed());
//         assert_eq!(
//             track.bbox().to_tlwh(),
//             arr1::<f32>(&[0.9682312, 0.8316479, 8.2856045, 8.30345])
//         );

//         Ok(())
//     }
// }
