use std::rc::Rc;

use crate::*;

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
/// let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, None, None, None);
/// let mut tracker = Tracker::new(metric, None, None, None);
///
/// // create a detection
/// // feature would be the feature vector from the cosine metric learning model output
/// let detection = Detection::new(
///     BoundingBox::new(0.0, 0.0, 5.0, 5.0),
///     1.0,
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
///         "{} {:?} {:?}",
///         track.track_id(),
///         track.state(),
///         track.to_tlwh(),
///     );
/// };
///```
#[derive(Debug)]
pub struct Tracker {
    /// The distance metric used for measurement to track association.
    metric: NearestNeighborDistanceMetric,
    /// Gating threshold for intersection over union. Associations with cost larger than this value are disregarded.
    max_iou_distance: f32,
    /// Maximum number of missed misses before a track is deleted.
    max_age: usize,
    /// Number of frames that a track remains in initialization phase.
    n_init: usize,
    /// A Kalman filter to filter target trajectories in image space.
    kf: KalmanFilter,
    /// The list of active tracks at the current time step.
    tracks: Vec<Track>,
    /// Used to allocate identifiers to new tracks.
    next_id: usize,
}

impl Tracker {
    /// Returns a new Tracker
    ///
    /// # Parameters
    ///
    /// - `metric`: A distance metric for measurement-to-track association.
    /// - `max_iou_distance`: Gating threshold for intersection over union. Associations with cost larger than this value are disregarded. Default `0.7`.
    /// - `max_age`: Maximum number of missed misses before a track is deleted. Default `30`.
    /// - `n_init`: Number of consecutive detections before the track is confirmed. The track state is set to `Deleted` if a miss occurs within the first `n_init` frames. Default `3`.
    /// - `feature_length`: The length of the feature vector used for
    pub fn new(
        metric: NearestNeighborDistanceMetric,
        max_iou_distance: Option<f32>,
        max_age: Option<usize>,
        n_init: Option<usize>,
    ) -> Tracker {
        Tracker {
            metric,
            max_iou_distance: max_iou_distance.unwrap_or(0.7),
            max_age: max_age.unwrap_or(30),
            n_init: n_init.unwrap_or(3),
            kf: KalmanFilter::new(),
            tracks: vec![],
            next_id: 1,
        }
    }

    /// Return the tracks
    pub fn tracks(&self) -> &Vec<Track> {
        &self.tracks
    }

    /// Propagate track state distributions one time step forward. This function should be called once every time step, before `update`.
    pub fn predict(&mut self) {
        let kf = self.kf.clone();
        self.tracks.iter_mut().for_each(|track| track.predict(&kf));
    }

    /// Perform measurement update and track management.
    ///
    /// # Parameters
    ///
    /// - `detections`: A list of detections at the current time step.
    pub fn update(&mut self, detections: &[Detection]) {
        // Run matching cascade.
        let (matches, unmatched_tracks, unmatched_detections) = self.match_impl(detections);

        // Update track set.
        for m in matches {
            self.tracks
                .get_mut(m.track_idx())
                .unwrap()
                .update(&self.kf, detections.get(m.detection_idx()).unwrap());
        }
        for unmatched_track in unmatched_tracks {
            self.tracks.get_mut(unmatched_track).unwrap().mark_missed();
        }
        for detection_idx in unmatched_detections {
            self.initiate_track(detections.get(detection_idx).unwrap().to_owned());
        }
        self.tracks.retain(|track| !track.is_deleted());

        // Update distance metric.
        let active_targets: Vec<usize> = self
            .tracks
            .iter()
            .filter(|track| track.is_confirmed())
            .map(|track| *track.track_id())
            .collect();

        let feature_length = *self.metric.feature_length();
        let mut features = Array2::<f32>::zeros((0, feature_length));
        let mut targets: Vec<usize> = vec![];
        self.tracks
            .iter_mut()
            .filter(|track| track.is_confirmed())
            .for_each(|track| {
                features = concatenate![Axis(0), features, *track.features()];
                for _ in 0..track.features().nrows() {
                    targets.push(*track.track_id());
                }
                *track.features_mut() = Array2::zeros((0, feature_length));
            });

        self.metric
            .partial_fit(&features, &targets, &active_targets)
    }

    fn match_impl(&self, detections: &[Detection]) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
        let metric = self.metric.clone();
        let kf = self.kf.clone();
        let feature_length = *self.metric.feature_length();

        let gated_metric = Rc::new(
            move |tracks: &[Track],
                  dets: &[Detection],
                  track_indices: Option<Vec<usize>>,
                  detection_indices: Option<Vec<usize>>|
                  -> Array2<f32> {
                let detection_indices = detection_indices.unwrap();
                let track_indices = track_indices.unwrap();

                let mut features = Array2::<f32>::zeros((0, feature_length));
                detection_indices.iter().for_each(|i| {
                    if let Some(feature) = dets.get(*i).unwrap().feature() {
                        features.push_row(feature.view()).unwrap()
                    }
                });
                let targets = track_indices
                    .iter()
                    .map(|i| *tracks.get(*i).unwrap().track_id())
                    .collect::<Vec<usize>>();

                let cost_matrix = metric.distance(&features, &detection_indices, &targets);

                linear_assignment::gate_cost_matrix(
                    kf.clone(),
                    cost_matrix,
                    tracks,
                    dets,
                    track_indices,
                    detection_indices,
                    None,
                    None,
                )
            },
        );

        // Split track set into confirmed and unconfirmed tracks.
        let confirmed_tracks: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, track)| track.is_confirmed())
            .map(|(i, _)| i)
            .collect();
        let unconfirmed_tracks: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, track)| !track.is_confirmed())
            .map(|(i, _)| i)
            .collect();

        // Associate confirmed tracks using appearance features.
        let (matches_a, unmatched_tracks_a, unmatched_detections) =
            linear_assignment::matching_cascade(
                gated_metric,
                *self.metric.matching_threshold(),
                self.max_age,
                &self.tracks,
                &detections,
                Some(confirmed_tracks),
                None,
            );

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        let iou_track_candidates = [
            unconfirmed_tracks,
            unmatched_tracks_a
                .iter()
                .filter(|k| {
                    let track = self.tracks.get(**k).unwrap();
                    *track.time_since_update() == 0 || track.features().nrows() == 0
                })
                .map(|v| v.to_owned())
                .collect::<Vec<usize>>(),
        ]
        .concat();

        let unmatched_tracks_a = unmatched_tracks_a
            .iter()
            .filter(|k| *self.tracks.get(**k).unwrap().time_since_update() != 0)
            .map(|v| v.to_owned())
            .collect::<Vec<usize>>();

        let (matches_b, unmatched_tracks_b, unmatched_detections) =
            linear_assignment::min_cost_matching(
                Rc::new(iou_matching::iou_cost),
                self.max_iou_distance,
                &self.tracks,
                detections,
                Some(iou_track_candidates),
                Some(unmatched_detections),
            );

        let matches = [matches_a, matches_b].concat();
        let mut unmatched_tracks = [unmatched_tracks_a, unmatched_tracks_b].concat();
        unmatched_tracks.dedup();

        (matches, unmatched_tracks, unmatched_detections)
    }

    fn initiate_track(&mut self, detection: Detection) {
        let (mean, covariance) = self.kf.initiate(&detection.to_xyah());
        let feature = detection
            .feature()
            .clone()
            .map(|feature| feature.insert_axis(Axis(0)));
        self.tracks.push(Track::new(
            mean,
            covariance,
            self.next_id,
            self.n_init,
            self.max_age,
            feature,
        ));
        self.next_id += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    use rand::prelude::*;
    use rand_distr::Normal;
    use rand_pcg::{Lcg64Xsh32, Pcg32};

    /// Returns a psuedo-random (deterministic) f32 between -0.5 and +0.5
    fn next_f32(rng: &mut Lcg64Xsh32) -> f32 {
        (rng.next_u32() as f64 / u32::MAX as f64) as f32 - 0.5
    }

    /// Returns a vec of length n with a normal distribution
    fn normal_vec(rng: &mut Lcg64Xsh32, mean: f32, std_dev: f32, n: i32) -> Vec<f32> {
        let normal = Normal::<f32>::new(mean, std_dev).unwrap();
        (0..n).map(|_| normal.sample(rng)).collect()
    }

    #[test]
    fn tracker() {
        let iterations: i32 = 100;
        let log = false;

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..8 * iterations)
            .map(|_| next_f32(&mut rng))
            .collect::<Vec<f32>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 8 * iterations);

        // create the feature vectors
        let d0_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d1_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d2_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d3_feat = normal_vec(&mut rng, 0.0, 1.0, 128);

        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, None, None, None);
        let mut tracker = Tracker::new(metric, None, None, None);

        for iteration in 0..iterations {
            // move up to right
            let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0 = Detection::new(
                BoundingBox::new(
                    d0_x,
                    d0_y,
                    10.0 + scale_jitter.pop().unwrap(),
                    10.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                Some(d0_feat.clone()),
            );

            // move down to left
            let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1 = Detection::new(
                BoundingBox::new(
                    d1_x,
                    d1_y,
                    8.0 + scale_jitter.pop().unwrap(),
                    8.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                Some(d1_feat.clone()),
            );

            // move up to left
            let d2_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d2_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d2 = Detection::new(
                BoundingBox::new(
                    d2_x,
                    d2_y,
                    6.0 + scale_jitter.pop().unwrap(),
                    6.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                Some(d2_feat.clone()),
            );

            // move up and right
            let d3_x = 0.0 + (iteration as f32 * 0.1) + movement_jitter.pop().unwrap();
            let d3_y = 0.0 + ((iteration - 50) as f32) + movement_jitter.pop().unwrap();
            let d3 = Detection::new(
                BoundingBox::new(
                    d3_x,
                    d3_y,
                    5.0 + scale_jitter.pop().unwrap(),
                    5.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                Some(d3_feat.clone()),
            );

            &tracker.predict();

            // add and remove detections over the sequence
            match iteration {
                _ if iteration >= 60 => {
                    &tracker.update(&[d2, d3]);
                }
                _ if iteration >= 30 => {
                    &tracker.update(&[d0, d2]);
                }
                _ => {
                    &tracker.update(&[d0, d1]);
                }
            }

            // for debugging
            if log {
                for track in &tracker.tracks {
                    println!(
                        "{}: {:?} {:?} {:?} {:?}",
                        iteration,
                        track.track_id(),
                        track.state(),
                        track.to_tlwh(),
                        tracker
                            .metric
                            .track_features(*track.track_id())
                            .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
                            .nrows(),
                    );
                }
            }
        }

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &3)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.to_tlwh(),
            arr1::<f32>(&[98.73315, 0.65894794, 5.728961, 5.717063])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &4)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.to_tlwh(),
            arr1::<f32>(&[10.129211, 48.62851, 5.1298714, 5.143466])
        );
    }

    #[test]
    fn tracker_no_features() {
        let iterations: i32 = 100;
        let log = true;

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..8 * iterations)
            .map(|_| next_f32(&mut rng))
            .collect::<Vec<f32>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 8 * iterations);

        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, None, None, None);
        let mut tracker = Tracker::new(metric, None, None, None);

        for iteration in 0..iterations {
            // move up to right
            let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0 = Detection::new(
                BoundingBox::new(
                    d0_x,
                    d0_y,
                    10.0 + scale_jitter.pop().unwrap(),
                    10.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
            );

            // move down to left
            let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1 = Detection::new(
                BoundingBox::new(
                    d1_x,
                    d1_y,
                    8.0 + scale_jitter.pop().unwrap(),
                    8.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
            );

            &tracker.predict();

            &tracker.update(&[d0, d1]);

            // for debugging
            if log {
                for track in &tracker.tracks {
                    println!(
                        "{}: {:?} {:?} {:?} {:?}",
                        iteration,
                        track.track_id(),
                        track.state(),
                        track.to_tlwh(),
                        tracker
                            .metric
                            .track_features(*track.track_id())
                            .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
                            .nrows(),
                    );
                }
            }
        }

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &1)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.to_tlwh(),
            arr1::<f32>(&[99.2418, 99.21735, 9.979219, 9.984485])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &2)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.to_tlwh(),
            arr1::<f32>(&[1.294354, 1.1528587, 8.003455, 8.0692625])
        );
    }
}
