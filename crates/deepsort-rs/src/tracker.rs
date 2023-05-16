use crate::*;
use anyhow::Result;
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
    /// The distance metric used for measurement to track association.
    nn_metric: NearestNeighborDistanceMetric,
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
    /// Confidence threshold for performing primary or secondary matching.
    track_threshold: f32,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new(
            NearestNeighborDistanceMetric::default(),
            None,
            None,
            None,
            None,
        )
    }
}

impl Tracker {
    /// Returns a new Tracker
    ///
    /// # Arguments
    ///
    /// * `nn_metric`: A distance metric for measurement-to-track association.
    /// * `max_iou_distance`: Gating threshold for intersection over union. Associations with cost larger than this value are disregarded. Default `0.7`.
    /// * `max_age`: Maximum number of missed misses before a track is deleted. Default `30`.
    /// * `n_init`: Number of consecutive detections before the track is confirmed. The track state is set to `Deleted` if a miss occurs within the first `n_init` frames. Default `3`.
    /// * `track_threshold`: Confidence threshold for performing primary or secondary matching. Default `0.6`.
    pub fn new(
        nn_metric: NearestNeighborDistanceMetric,
        max_iou_distance: Option<f32>,
        max_age: Option<usize>,
        n_init: Option<usize>,
        track_threshold: Option<f32>,
    ) -> Tracker {
        Tracker {
            nn_metric,
            max_iou_distance: max_iou_distance.unwrap_or(0.7),
            max_age: max_age.unwrap_or(30),
            n_init: n_init.unwrap_or(3),
            kf: KalmanFilter::default(),
            tracks: vec![],
            next_id: 1,
            track_threshold: track_threshold.unwrap_or(0.6),
        }
    }

    /// Set nn_metric
    pub fn with_nn_metric(&mut self, nn_metric: NearestNeighborDistanceMetric) -> &mut Self {
        self.nn_metric = nn_metric;
        self
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

    /// Set n_init
    pub fn with_n_init(&mut self, n_init: usize) -> &mut Self {
        self.n_init = n_init;
        self
    }

    /// Return the tracks
    pub fn tracks(&self) -> &Vec<Track> {
        &self.tracks
    }

    /// Return the nn_metric
    pub fn nn_metric(&self) -> &NearestNeighborDistanceMetric {
        &self.nn_metric
    }

    /// Propagate track state distributions one time step forward. This function should be called once every time step, before `update`.
    pub fn predict(&mut self) {
        self.tracks
            .iter_mut()
            .for_each(|track| track.predict(&self.kf));
    }

    /// Perform measurement update and track management.
    ///
    /// # Parameters
    ///
    /// - `detections`: A list of detections at the current time step.
    pub fn update(&mut self, detections: &[Detection]) -> Result<()> {
        // Run matching cascade.
        let (features_matches, iou_matches, unmatched_tracks, unmatched_detections) =
            self.matching_cascade(detections)?;

        // Update track set.
        for feature_match in features_matches {
            let detection = detections.get(feature_match.detection_idx()).unwrap();
            let track = self.tracks.get_mut(feature_match.track_idx()).unwrap();
            track.update(
                &self.kf,
                detection,
                Some(MatchSource::NearestNeighbor {
                    detection: detection.clone(),
                    distance: feature_match.distance(),
                }),
            )?;
        }
        for iou_match in iou_matches {
            let detection = detections.get(iou_match.detection_idx()).unwrap();
            let track = self.tracks.get_mut(iou_match.track_idx()).unwrap();
            track.update(
                &self.kf,
                detection,
                Some(MatchSource::IoU {
                    detection: detection.clone(),
                    distance: iou_match.distance(),
                }),
            )?;
        }
        for unmatched_track in unmatched_tracks {
            self.tracks.get_mut(unmatched_track).unwrap().mark_missed();
        }
        for detection_idx in unmatched_detections {
            self.initiate_track(detections.get(detection_idx).unwrap().to_owned());
        }

        // remove deleted tracks
        self.tracks.retain(|track| !track.is_deleted());

        // Update distance metric.
        let active_targets: Vec<usize> = self
            .tracks
            .iter()
            .filter_map(|track| track.is_confirmed().then(|| track.track_id()))
            .collect();

        // For any confirmed tracks that have features 'partial_fit' the features into the metric.samples hashmap and remove from track
        let tracks_with_features = self
            .tracks
            .iter_mut()
            .filter(|track| {
                track.is_confirmed()
                    && track
                        .features()
                        .map(|features| features.nrows() != 0)
                        .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        if !tracks_with_features.is_empty() {
            let mut targets: Vec<usize> = vec![];
            let features = concatenate(
                Axis(0),
                &tracks_with_features
                    .iter()
                    .map(|track| {
                        let features = track.features().take().unwrap();
                        for _ in 0..features.nrows() {
                            targets.push(track.track_id());
                        }
                        features.view()
                    })
                    .collect::<Vec<_>>(),
            )?;

            self.nn_metric
                .partial_fit(&features, &targets, &active_targets)?;
        }

        Ok(())
    }

    /// The matching cascade.
    ///
    /// It works in two stages:
    /// - first run the nn_metric matching to try to associate matches using the feature vector
    /// - with the remaining tracks attempt to match using iou
    #[allow(clippy::type_complexity)]
    fn matching_cascade(
        &self,
        detections: &[Detection],
    ) -> Result<(Vec<Match>, Vec<Match>, Vec<usize>, Vec<usize>)> {
        // Split track set into confirmed and unconfirmed tracks.
        let mut confirmed_tracks = Vec::new();
        let mut unconfirmed_tracks = Vec::new();
        self.tracks.iter().enumerate().for_each(|(i, track)| {
            if track.is_confirmed() {
                confirmed_tracks.push(i);
            } else {
                unconfirmed_tracks.push(i);
            }
        });

        // Associate only confirmed tracks using appearance features.
        let (features_matches, features_unmatched_tracks, unmatched_detections) =
            linear_assignment::matching_cascade(
                self.nn_metric.distance_metric(&self.kf),
                self.nn_metric.matching_threshold(),
                self.max_age,
                &self.tracks,
                detections,
                Some(confirmed_tracks),
                None,
            )?;

        // partition the unmatched tracks into recent (time_since_update == 1) and older
        let (features_unmatched_tracks_recent, features_unmatched_tracks_older): (
            Vec<usize>,
            Vec<usize>,
        ) = features_unmatched_tracks
            .into_iter()
            .partition(|k| self.tracks.get(*k).unwrap().time_since_update() == 1);

        // Associate recent tracks together with unconfirmed tracks using IOU.
        let iou_track_candidates = [unconfirmed_tracks, features_unmatched_tracks_recent].concat();
        let (iou_matches, iou_unmatched_tracks, unmatched_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::intersection_over_union_cost(),
                self.max_iou_distance,
                &self.tracks,
                detections,
                Some(iou_track_candidates),
                Some(unmatched_detections),
            )?;

        let mut unmatched_tracks = [features_unmatched_tracks_older, iou_unmatched_tracks].concat();
        unmatched_tracks.dedup();

        Ok((
            features_matches,
            iou_matches,
            unmatched_tracks,
            unmatched_detections,
        ))
    }

    // /// The matching cascade.
    // ///
    // /// It works in two stages:
    // /// - first run the nn_metric matching to try to associate matches using the feature vector
    // /// - with the remaining tracks attempt to match using iou
    // #[allow(clippy::type_complexity)]
    // fn matching_cascade(
    //     &self,
    //     detections: &[Detection],
    // ) -> Result<(Vec<Match>, Vec<Match>, Vec<usize>, Vec<usize>)> {
    //     // Split track set into confirmed and unconfirmed tracks.
    //     let mut confirmed_tracks = Vec::new();
    //     let mut unconfirmed_tracks = Vec::new();
    //     self.tracks.iter().enumerate().for_each(|(i, track)| {
    //         if track.is_confirmed() {
    //             confirmed_tracks.push(i);
    //         } else {
    //             unconfirmed_tracks.push(i);
    //         }
    //     });

    //     let mut high_detections = Vec::new();
    //     let mut low_detections = Vec::new();
    //     detections.iter().enumerate().for_each(|(i, detection)| {
    //         if detection.confidence() > self.track_threshold {
    //             high_detections.push(i);
    //         } else {
    //             low_detections.push(i);
    //         }
    //     });

    //     // Associate only confirmed tracks using appearance features.
    //     let (features_matches, features_unmatched_tracks, features_unmatched_high_detections) =
    //         linear_assignment::matching_cascade(
    //             self.nn_metric.distance_metric(&self.kf),
    //             self.nn_metric.matching_threshold(),
    //             self.max_age,
    //             &self.tracks,
    //             detections,
    //             Some(confirmed_tracks),
    //             Some(high_detections),
    //         )?;

    //     // partition the unmatched tracks into recent (time_since_update == 1) and older
    //     let (features_unmatched_tracks_recent, features_unmatched_tracks_older): (
    //         Vec<usize>,
    //         Vec<usize>,
    //     ) = features_unmatched_tracks
    //         .into_iter()
    //         .partition(|k| self.tracks.get(*k).unwrap().time_since_update() == 1);

    //     // Associate recent tracks together with unconfirmed tracks using IOU.
    //     let iou_track_candidates = [unconfirmed_tracks, features_unmatched_tracks_recent].concat();
    //     let (high_iou_matches, iou_unmatched_tracks, unmatched_high_detections) =
    //         linear_assignment::min_cost_matching(
    //             iou_matching::intersection_over_union_cost(),
    //             self.max_iou_distance,
    //             &self.tracks,
    //             detections,
    //             Some(iou_track_candidates),
    //             Some(features_unmatched_high_detections),
    //         )?;

    //     // Associate recent tracks together with unconfirmed tracks using IOU.
    //     let iou_track_candidates = iou_unmatched_tracks;
    //     let (low_iou_matches, iou_unmatched_tracks, unmatched_low_detections) =
    //         linear_assignment::min_cost_matching(
    //             iou_matching::intersection_over_union_cost(),
    //             self.max_iou_distance,
    //             &self.tracks,
    //             detections,
    //             Some(iou_track_candidates),
    //             Some(low_detections),
    //         )?;

    //     let mut unmatched_tracks = [features_unmatched_tracks_older, iou_unmatched_tracks].concat();
    //     unmatched_tracks.dedup();

    //     Ok((
    //         features_matches,
    //         [high_iou_matches, low_iou_matches].concat(),
    //         unmatched_tracks,
    //         [unmatched_high_detections, unmatched_low_detections].concat(),
    //     ))
    // }

    fn initiate_track(&mut self, detection: Detection) {
        let (mean, covariance) = self.kf.initiate(detection.bbox());
        let features = detection
            .feature()
            .as_ref()
            .map(|feature| feature.clone().insert_axis(Axis(0)));
        self.tracks.push(Track::new(
            mean,
            covariance,
            self.next_id,
            detection,
            self.n_init,
            self.max_age,
            features,
        ));
        self.next_id += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::{Ok, Result};
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
    fn tracker_nearest_neighbor() -> Result<()> {
        let iterations: i32 = 100;

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<_>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

        // create the feature vectors
        let d0_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d1_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d2_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d3_feat = normal_vec(&mut rng, 0.0, 1.0, 128);

        let mut tracker = Tracker::default();

        for iteration in 0..iterations {
            // move up to right
            let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0 = Detection::new(
                None,
                BoundingBox::new(
                    d0_x,
                    d0_y,
                    10.0 + scale_jitter.pop().unwrap(),
                    10.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                Some(d0_feat.clone()),
            );

            // move down to left
            let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1 = Detection::new(
                None,
                BoundingBox::new(
                    d1_x,
                    d1_y,
                    8.0 + scale_jitter.pop().unwrap(),
                    8.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                Some(d1_feat.clone()),
            );

            // move up to left
            let d2_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d2_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d2 = Detection::new(
                None,
                BoundingBox::new(
                    d2_x,
                    d2_y,
                    6.0 + scale_jitter.pop().unwrap(),
                    6.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                Some(d2_feat.clone()),
            );

            // move up and right
            let d3_x = 0.0 + (iteration as f32 * 0.1) + movement_jitter.pop().unwrap();
            let d3_y = 0.0 + ((iteration - 50) as f32) + movement_jitter.pop().unwrap();
            let d3 = Detection::new(
                None,
                BoundingBox::new(
                    d3_x,
                    d3_y,
                    5.0 + scale_jitter.pop().unwrap(),
                    5.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                Some(d3_feat.clone()),
            );

            tracker.predict();

            // add and remove detections over the sequence
            match iteration {
                _ if iteration >= 60 => {
                    tracker.update(&[d2, d3])?;
                }
                _ if iteration >= 30 => {
                    tracker.update(&[d0, d2])?;
                }
                _ => {
                    tracker.update(&[d0, d1])?;
                }
            }

            // for debugging
            // for track in &tracker.tracks {
            //     println!(
            //         "{}: {:?} {:?} {:?} {:?} {:?}",
            //         iteration,
            //         track.track_id(),
            //         track.state(),
            //         track.bbox().to_tlwh(),
            //         tracker
            //             .metric
            //             .track_features(*track.track_id())
            //             .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
            //             .nrows(),
            //         track.match_source(),
            //     );
            // }
        }

        assert_eq!(tracker.tracks.len(), 2);

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == 3)
            .collect::<Vec<_>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[99.12867, 1.0377614, 6.1343956, 6.1184144])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == 4)
            .collect::<Vec<_>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[9.640262, 48.84033, 5.1905212, 5.113961])
        );

        Ok(())
    }

    #[test]
    fn tracker_intersection_over_union() -> Result<()> {
        let iterations: i32 = 100;

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<_>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

        let mut tracker = Tracker::default();

        for iteration in 0..iterations {
            // move up to right
            let d0_x = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0_y = 0.0 + (iteration as f32) + movement_jitter.pop().unwrap();
            let d0 = Detection::new(
                None,
                BoundingBox::new(
                    d0_x,
                    d0_y,
                    10.0 + scale_jitter.pop().unwrap(),
                    10.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                None,
            );

            // move down to left
            let d1_x = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1_y = 100.0 - (iteration as f32) + movement_jitter.pop().unwrap();
            let d1 = Detection::new(
                None,
                BoundingBox::new(
                    d1_x,
                    d1_y,
                    8.0 + scale_jitter.pop().unwrap(),
                    8.0 + scale_jitter.pop().unwrap(),
                ),
                1.0,
                None,
                None,
                None,
            );

            tracker.predict();
            tracker.update(&[d0, d1])?;

            // for debugging
            // for track in &tracker.tracks {
            //     println!(
            //         "{}: {:?} {:?} {:?} {:?} {:?}",
            //         iteration,
            //         track.track_id(),
            //         track.state(),
            //         track.bbox().to_tlwh(),
            //         tracker
            //             .metric
            //             .track_features(*track.track_id())
            //             .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
            //             .nrows(),
            //         track.match_source(),
            //     );
            // }
        }

        assert_eq!(tracker.tracks.len(), 2);

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == 1)
            .collect::<Vec<_>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[99.26583, 98.69391, 9.864473, 9.7438])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == 2)
            .collect::<Vec<_>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[0.9682312, 0.8316479, 8.2856045, 8.30345])
        );

        Ok(())
    }
}
