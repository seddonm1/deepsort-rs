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
///         track.confidence(),
///         track.state(),
///         track.bbox().to_tlwh(),
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

    /// Return the metric
    pub fn metric(&self) -> &NearestNeighborDistanceMetric {
        &self.metric
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
        let (features_matches, iou_matches, unmatched_tracks, unmatched_detections) =
            self.match_impl(detections);

        // Update track set.
        for m in features_matches {
            let track = self.tracks.get_mut(*m.track_idx()).unwrap();
            track.update(
                &self.kf,
                detections.get(*m.detection_idx()).unwrap(),
                Some(MatchSource::NearestNeighbor {
                    distance: *m.distance(),
                }),
            );
        }
        for m in iou_matches {
            let track = self.tracks.get_mut(*m.track_idx()).unwrap();
            track.update(
                &self.kf,
                detections.get(*m.detection_idx()).unwrap(),
                Some(MatchSource::IoU {
                    distance: *m.distance(),
                }),
            );
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

        // For any confirmed tracks 'partial_fit' the features into the metric.samples hashmap and remove from track
        let feature_length = *self.metric.feature_length();
        let mut features = Array2::<f32>::zeros((0, feature_length));
        let mut targets: Vec<usize> = vec![];
        self.tracks
            .iter_mut()
            .filter(|track| track.is_confirmed() && track.features().nrows() != 0)
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

    fn match_impl(
        &self,
        detections: &[Detection],
    ) -> (Vec<Match>, Vec<Match>, Vec<usize>, Vec<usize>) {
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
        let (features_matches, features_unmatched_tracks, unmatched_detections) =
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
            features_unmatched_tracks
                .iter()
                .filter(|k| *self.tracks.get(**k).unwrap().time_since_update() == 1)
                .map(|v| v.to_owned())
                .collect::<Vec<usize>>(),
        ]
        .concat();

        let features_unmatched_tracks = features_unmatched_tracks
            .iter()
            .filter(|k| *self.tracks.get(**k).unwrap().time_since_update() != 1)
            .map(|v| v.to_owned())
            .collect::<Vec<usize>>();

        let (iou_matches, iou_unmatched_tracks, unmatched_detections) =
            linear_assignment::min_cost_matching(
                Rc::new(iou_matching::iou_cost),
                self.max_iou_distance,
                &self.tracks,
                detections,
                Some(iou_track_candidates),
                Some(unmatched_detections),
            );

        let mut unmatched_tracks = [features_unmatched_tracks, iou_unmatched_tracks].concat();
        unmatched_tracks.dedup();

        (
            features_matches,
            iou_matches,
            unmatched_tracks,
            unmatched_detections,
        )
    }

    fn initiate_track(&mut self, detection: Detection) {
        let (mean, covariance) = self.kf.initiate(&detection.bbox());
        let feature = detection
            .feature()
            .clone()
            .map(|feature| feature.insert_axis(Axis(0)));
        self.tracks.push(Track::new(
            mean,
            covariance,
            self.next_id,
            *detection.confidence(),
            *detection.class_id(),
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

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<f32>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

        // create the feature vectors
        let d0_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d1_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d2_feat = normal_vec(&mut rng, 0.0, 1.0, 128);
        let d3_feat = normal_vec(&mut rng, 0.0, 1.0, 128);

        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, None, None, None);
        let mut tracker = Tracker::new(metric, None, None, None);

        for iteration in 0..iterations {
            println!("\n{}", iteration);
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
                None,
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
                None,
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
                None,
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
            for track in &tracker.tracks {
                println!(
                    "{}: {:?} {:?} {:?} {:?} {:?}",
                    iteration,
                    track.track_id(),
                    track.state(),
                    track.bbox().to_tlwh(),
                    tracker
                        .metric
                        .track_features(*track.track_id())
                        .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
                        .nrows(),
                    track.match_source(),
                );
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
            track.bbox().to_tlwh(),
            arr1::<f32>(&[99.12867, 1.0377614, 6.1343956, 6.1184144])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &4)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[9.640262, 48.84033, 5.1905212, 5.113961])
        );
    }

    #[test]
    fn tracker_no_features() {
        let iterations: i32 = 100;

        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);

        // create random movement/scale this is a vectors so it can be easily copied to python for comparison
        let mut movement_jitter = (0..1000).map(|_| next_f32(&mut rng)).collect::<Vec<f32>>();
        let mut scale_jitter = normal_vec(&mut rng, 0.0, 0.2, 1000);

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
                None,
            );

            &tracker.predict();
            &tracker.update(&[d0, d1]);

            // for debugging
            for track in &tracker.tracks {
                println!(
                    "{}: {:?} {:?} {:?} {:?} {:?}",
                    iteration,
                    track.track_id(),
                    track.state(),
                    track.bbox().to_tlwh(),
                    tracker
                        .metric
                        .track_features(*track.track_id())
                        .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
                        .nrows(),
                    track.match_source(),
                );
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
            track.bbox().to_tlwh(),
            arr1::<f32>(&[99.26583, 98.69391, 9.864473, 9.7438])
        );

        let track = tracker
            .tracks
            .iter()
            .filter(|track| track.track_id() == &2)
            .collect::<Vec<&Track>>();
        let track = track.first().unwrap();
        assert!(track.is_confirmed());
        assert_eq!(
            track.bbox().to_tlwh(),
            arr1::<f32>(&[0.9682312, 0.8316479, 8.2856045, 8.30345])
        );
    }

    #[test]
    fn tracker_debug() {
        let mut detections: Vec<Vec<Vec<f32>>> = vec![
            vec![
                vec![1452.0, 32.0, 79.0, 155.0],
                vec![1648.0, 119.0, 68.0, 170.0],
                vec![801.0, 243.0, 55.0, 179.0],
                vec![1604.0, 611.0, 112.0, 251.0],
                vec![248.0, 787.0, 135.0, 293.0],
                vec![880.0, 77.0, 62.0, 154.0],
                vec![1349.0, 16.0, 52.0, 115.0],
                vec![1570.0, 910.0, 153.0, 170.0],
                vec![1322.0, 0.0, 41.0, 50.0],
                vec![712.0, 227.0, 73.0, 183.0],
                vec![987.0, 2.0, 45.0, 124.0],
                vec![1154.0, 0.0, 42.0, 76.0],
                vec![1798.0, 175.0, 67.0, 198.0],
                vec![287.0, 297.0, 83.0, 184.0],
                vec![865.0, 394.0, 120.0, 193.0],
                vec![1232.0, 0.0, 30.0, 15.0],
            ],
            vec![
                vec![1455.0, 31.0, 76.0, 156.0],
                vec![1604.0, 612.0, 111.0, 250.0],
                vec![804.0, 241.0, 55.0, 179.0],
                vec![1648.0, 119.0, 67.0, 169.0],
                vec![1571.0, 913.0, 148.0, 167.0],
                vec![242.0, 788.0, 123.0, 292.0],
                vec![862.0, 396.0, 113.0, 185.0],
                vec![882.0, 74.0, 61.0, 158.0],
                vec![1352.0, 14.0, 53.0, 116.0],
                vec![988.0, 1.0, 46.0, 125.0],
                vec![1323.0, 0.0, 41.0, 50.0],
                vec![1154.0, 0.0, 41.0, 77.0],
                vec![1800.0, 175.0, 67.0, 181.0],
                vec![719.0, 221.0, 69.0, 188.0],
                vec![1231.0, 0.0, 35.0, 15.0],
                vec![282.0, 298.0, 77.0, 185.0],
            ],
            vec![
                vec![1455.0, 31.0, 76.0, 156.0],
                vec![1604.0, 613.0, 111.0, 249.0],
                vec![804.0, 241.0, 55.0, 179.0],
                vec![1648.0, 119.0, 67.0, 169.0],
                vec![1572.0, 913.0, 147.0, 167.0],
                vec![239.0, 786.0, 131.0, 293.0],
                vec![862.0, 396.0, 112.0, 186.0],
                vec![882.0, 74.0, 61.0, 159.0],
                vec![988.0, 1.0, 46.0, 125.0],
                vec![1352.0, 14.0, 53.0, 117.0],
                vec![1323.0, 0.0, 41.0, 49.0],
                vec![1154.0, 0.0, 41.0, 77.0],
                vec![1800.0, 175.0, 67.0, 181.0],
                vec![719.0, 221.0, 68.0, 187.0],
                vec![1231.0, 0.0, 35.0, 15.0],
                vec![282.0, 298.0, 76.0, 185.0],
            ],
            vec![
                vec![1455.0, 31.0, 76.0, 156.0],
                vec![1604.0, 613.0, 111.0, 249.0],
                vec![804.0, 241.0, 55.0, 179.0],
                vec![1648.0, 119.0, 67.0, 169.0],
                vec![1572.0, 913.0, 147.0, 167.0],
                vec![239.0, 786.0, 131.0, 293.0],
                vec![862.0, 396.0, 112.0, 186.0],
                vec![882.0, 74.0, 61.0, 158.0],
                vec![988.0, 1.0, 46.0, 125.0],
                vec![1352.0, 14.0, 53.0, 117.0],
                vec![1323.0, 0.0, 41.0, 49.0],
                vec![1154.0, 0.0, 41.0, 77.0],
                vec![1800.0, 175.0, 67.0, 181.0],
                vec![719.0, 221.0, 68.0, 187.0],
                vec![1231.0, 0.0, 35.0, 15.0],
                vec![282.0, 298.0, 77.0, 185.0],
            ],
            vec![
                vec![1601.0, 613.0, 114.0, 248.0],
                vec![1456.0, 30.0, 78.0, 156.0],
                vec![1648.0, 119.0, 67.0, 163.0],
                vec![807.0, 240.0, 58.0, 180.0],
                vec![1573.0, 917.0, 148.0, 163.0],
                vec![230.0, 792.0, 125.0, 288.0],
                vec![861.0, 395.0, 110.0, 210.0],
                vec![1800.0, 174.0, 70.0, 176.0],
                vec![1322.0, 0.0, 39.0, 50.0],
                vec![883.0, 74.0, 60.0, 151.0],
                vec![1357.0, 13.0, 52.0, 115.0],
                vec![989.0, 0.0, 47.0, 125.0],
                vec![1155.0, 0.0, 38.0, 77.0],
                vec![721.0, 219.0, 68.0, 188.0],
                vec![276.0, 296.0, 79.0, 189.0],
                vec![1229.0, 0.0, 42.0, 15.0],
                vec![299.0, 236.0, 82.0, 177.0],
                vec![1833.0, 275.0, 44.0, 69.0],
            ],
            vec![
                vec![1599.0, 614.0, 116.0, 248.0],
                vec![1458.0, 30.0, 76.0, 156.0],
                vec![223.0, 794.0, 116.0, 286.0],
                vec![1648.0, 119.0, 66.0, 163.0],
                vec![858.0, 398.0, 106.0, 227.0],
                vec![1579.0, 924.0, 141.0, 156.0],
                vec![811.0, 240.0, 66.0, 180.0],
                vec![884.0, 71.0, 60.0, 158.0],
                vec![1322.0, 0.0, 39.0, 48.0],
                vec![1154.0, 0.0, 40.0, 77.0],
                vec![1361.0, 11.0, 52.0, 105.0],
                vec![1800.0, 178.0, 71.0, 162.0],
                vec![988.0, 0.0, 51.0, 126.0],
                vec![267.0, 299.0, 85.0, 190.0],
                vec![724.0, 218.0, 73.0, 190.0],
                vec![305.0, 235.0, 73.0, 179.0],
                vec![632.0, 992.0, 196.0, 88.0],
                vec![1230.0, 0.0, 45.0, 14.0],
            ],
            vec![
                vec![1599.0, 617.0, 117.0, 246.0],
                vec![852.0, 399.0, 108.0, 233.0],
                vec![1460.0, 29.0, 74.0, 158.0],
                vec![1648.0, 118.0, 66.0, 164.0],
                vec![812.0, 240.0, 71.0, 179.0],
                vec![887.0, 71.0, 60.0, 156.0],
                vec![1366.0, 11.0, 52.0, 104.0],
                vec![1323.0, 0.0, 38.0, 48.0],
                vec![1152.0, 0.0, 41.0, 77.0],
                vec![725.0, 217.0, 74.0, 188.0],
                vec![1582.0, 935.0, 138.0, 145.0],
                vec![262.0, 306.0, 90.0, 185.0],
                vec![214.0, 801.0, 121.0, 279.0],
                vec![1800.0, 176.0, 73.0, 155.0],
                vec![989.0, 0.0, 53.0, 124.0],
                vec![308.0, 236.0, 68.0, 176.0],
                vec![641.0, 989.0, 176.0, 91.0],
                vec![1832.0, 280.0, 46.0, 60.0],
                vec![1231.0, 0.0, 44.0, 14.0],
            ],
            vec![
                vec![1599.0, 623.0, 118.0, 248.0],
                vec![1649.0, 116.0, 65.0, 165.0],
                vec![1457.0, 30.0, 80.0, 156.0],
                vec![812.0, 239.0, 71.0, 178.0],
                vec![888.0, 72.0, 62.0, 155.0],
                vec![1369.0, 7.0, 55.0, 106.0],
                vec![1152.0, 0.0, 41.0, 76.0],
                vec![1322.0, 0.0, 38.0, 47.0],
                vec![844.0, 405.0, 120.0, 236.0],
                vec![727.0, 217.0, 76.0, 187.0],
                vec![209.0, 817.0, 128.0, 262.0],
                vec![1802.0, 179.0, 75.0, 160.0],
                vec![991.0, 0.0, 52.0, 124.0],
                vec![260.0, 313.0, 89.0, 174.0],
                vec![309.0, 226.0, 69.0, 184.0],
                vec![647.0, 985.0, 156.0, 95.0],
                vec![1575.0, 938.0, 147.0, 142.0],
            ],
            vec![
                vec![1601.0, 630.0, 117.0, 242.0],
                vec![1649.0, 115.0, 65.0, 164.0],
                vec![1458.0, 30.0, 79.0, 156.0],
                vec![815.0, 236.0, 70.0, 180.0],
                vec![891.0, 72.0, 60.0, 154.0],
                vec![732.0, 217.0, 74.0, 189.0],
                vec![1150.0, 0.0, 42.0, 77.0],
                vec![835.0, 411.0, 121.0, 228.0],
                vec![1321.0, 0.0, 39.0, 47.0],
                vec![1802.0, 178.0, 76.0, 135.0],
                vec![204.0, 827.0, 133.0, 252.0],
                vec![1374.0, 7.0, 54.0, 103.0],
                vec![650.0, 979.0, 154.0, 101.0],
                vec![253.0, 318.0, 96.0, 181.0],
                vec![991.0, 0.0, 52.0, 125.0],
                vec![309.0, 234.0, 70.0, 177.0],
                vec![1828.0, 284.0, 49.0, 57.0],
                vec![1583.0, 950.0, 131.0, 130.0],
                vec![1236.0, 0.0, 37.0, 13.0],
            ],
            vec![
                vec![1601.0, 630.0, 117.0, 241.0],
                vec![1649.0, 115.0, 65.0, 164.0],
                vec![1458.0, 30.0, 79.0, 156.0],
                vec![815.0, 236.0, 71.0, 180.0],
                vec![891.0, 72.0, 60.0, 154.0],
                vec![732.0, 218.0, 74.0, 188.0],
                vec![836.0, 411.0, 120.0, 228.0],
                vec![1150.0, 0.0, 42.0, 77.0],
                vec![1321.0, 0.0, 39.0, 47.0],
                vec![1802.0, 178.0, 76.0, 134.0],
                vec![204.0, 827.0, 132.0, 252.0],
                vec![1374.0, 7.0, 54.0, 103.0],
                vec![650.0, 980.0, 153.0, 100.0],
                vec![253.0, 318.0, 96.0, 180.0],
                vec![991.0, 0.0, 52.0, 125.0],
                vec![309.0, 234.0, 70.0, 177.0],
                vec![1828.0, 284.0, 49.0, 57.0],
                vec![1583.0, 950.0, 132.0, 130.0],
                vec![1236.0, 0.0, 38.0, 13.0],
            ],
            vec![
                vec![1600.0, 635.0, 118.0, 232.0],
                vec![1649.0, 114.0, 65.0, 161.0],
                vec![1459.0, 29.0, 77.0, 157.0],
                vec![834.0, 413.0, 115.0, 234.0],
                vec![893.0, 71.0, 61.0, 154.0],
                vec![818.0, 234.0, 67.0, 182.0],
                vec![199.0, 833.0, 137.0, 246.0],
                vec![1803.0, 177.0, 79.0, 133.0],
                vec![1378.0, 4.0, 54.0, 112.0],
                vec![1321.0, 0.0, 41.0, 46.0],
                vec![737.0, 216.0, 72.0, 190.0],
                vec![1150.0, 0.0, 42.0, 79.0],
                vec![316.0, 223.0, 66.0, 189.0],
                vec![650.0, 976.0, 155.0, 103.0],
                vec![247.0, 320.0, 94.0, 179.0],
                vec![993.0, 0.0, 50.0, 126.0],
                vec![1237.0, 0.0, 37.0, 13.0],
                vec![1829.0, 284.0, 48.0, 57.0],
            ],
            vec![
                vec![1649.0, 113.0, 64.0, 157.0],
                vec![1463.0, 29.0, 75.0, 153.0],
                vec![1602.0, 640.0, 117.0, 225.0],
                vec![896.0, 71.0, 60.0, 151.0],
                vec![834.0, 418.0, 107.0, 232.0],
                vec![823.0, 233.0, 65.0, 179.0],
                vec![1805.0, 176.0, 79.0, 131.0],
                vec![193.0, 839.0, 129.0, 240.0],
                vec![738.0, 219.0, 74.0, 188.0],
                vec![1321.0, 0.0, 41.0, 46.0],
                vec![1385.0, 1.0, 52.0, 113.0],
                vec![1145.0, 0.0, 45.0, 80.0],
                vec![320.0, 224.0, 66.0, 186.0],
                vec![996.0, 0.0, 51.0, 124.0],
                vec![652.0, 974.0, 155.0, 105.0],
                vec![244.0, 320.0, 80.0, 178.0],
                vec![1827.0, 285.0, 50.0, 56.0],
                vec![1238.0, 0.0, 36.0, 12.0],
            ],
            vec![
                vec![1465.0, 27.0, 73.0, 154.0],
                vec![1650.0, 111.0, 63.0, 156.0],
                vec![1603.0, 654.0, 119.0, 221.0],
                vec![897.0, 70.0, 61.0, 153.0],
                vec![833.0, 420.0, 108.0, 230.0],
                vec![1387.0, 0.0, 51.0, 113.0],
                vec![1808.0, 174.0, 77.0, 132.0],
                vec![831.0, 231.0, 59.0, 176.0],
                vec![1145.0, 0.0, 44.0, 80.0],
                vec![997.0, 0.0, 51.0, 124.0],
                vec![1322.0, 0.0, 38.0, 46.0],
                vec![742.0, 217.0, 70.0, 188.0],
                vec![188.0, 844.0, 114.0, 236.0],
                vec![658.0, 966.0, 153.0, 113.0],
                vec![323.0, 224.0, 63.0, 183.0],
                vec![235.0, 321.0, 84.0, 179.0],
                vec![1826.0, 285.0, 51.0, 56.0],
                vec![1246.0, 0.0, 33.0, 11.0],
            ],
            vec![
                vec![1465.0, 27.0, 72.0, 154.0],
                vec![1650.0, 111.0, 63.0, 156.0],
                vec![1603.0, 654.0, 119.0, 221.0],
                vec![897.0, 69.0, 61.0, 154.0],
                vec![833.0, 420.0, 108.0, 230.0],
                vec![1387.0, 0.0, 51.0, 113.0],
                vec![1808.0, 174.0, 77.0, 133.0],
                vec![831.0, 230.0, 59.0, 177.0],
                vec![997.0, 0.0, 51.0, 123.0],
                vec![1144.0, 0.0, 45.0, 80.0],
                vec![1322.0, 0.0, 38.0, 46.0],
                vec![743.0, 217.0, 69.0, 188.0],
                vec![188.0, 844.0, 114.0, 236.0],
                vec![658.0, 966.0, 153.0, 113.0],
                vec![323.0, 224.0, 63.0, 183.0],
                vec![235.0, 321.0, 84.0, 179.0],
                vec![1826.0, 285.0, 51.0, 56.0],
                vec![1246.0, 0.0, 33.0, 11.0],
            ],
            vec![
                vec![1467.0, 26.0, 70.0, 154.0],
                vec![1649.0, 109.0, 64.0, 158.0],
                vec![835.0, 229.0, 55.0, 176.0],
                vec![1604.0, 653.0, 122.0, 221.0],
                vec![830.0, 422.0, 108.0, 229.0],
                vec![898.0, 71.0, 61.0, 151.0],
                vec![1000.0, 0.0, 51.0, 122.0],
                vec![1145.0, 0.0, 42.0, 82.0],
                vec![1391.0, 0.0, 50.0, 110.0],
                vec![1322.0, 0.0, 37.0, 46.0],
                vec![748.0, 220.0, 68.0, 177.0],
                vec![1812.0, 174.0, 74.0, 132.0],
                vec![183.0, 850.0, 105.0, 230.0],
                vec![669.0, 959.0, 144.0, 121.0],
                vec![330.0, 225.0, 66.0, 181.0],
                vec![1247.0, 0.0, 32.0, 10.0],
                vec![1825.0, 285.0, 52.0, 56.0],
                vec![229.0, 319.0, 74.0, 178.0],
            ],
            vec![
                vec![1467.0, 26.0, 70.0, 154.0],
                vec![1649.0, 109.0, 64.0, 158.0],
                vec![835.0, 229.0, 55.0, 176.0],
                vec![1604.0, 653.0, 122.0, 221.0],
                vec![831.0, 422.0, 107.0, 229.0],
                vec![898.0, 71.0, 60.0, 151.0],
                vec![1000.0, 0.0, 51.0, 122.0],
                vec![1145.0, 0.0, 42.0, 82.0],
                vec![1391.0, 0.0, 50.0, 110.0],
                vec![1322.0, 0.0, 37.0, 46.0],
                vec![1812.0, 173.0, 74.0, 134.0],
                vec![748.0, 220.0, 68.0, 178.0],
                vec![183.0, 847.0, 107.0, 233.0],
                vec![672.0, 960.0, 140.0, 120.0],
                vec![330.0, 225.0, 66.0, 181.0],
                vec![1247.0, 0.0, 32.0, 10.0],
                vec![1825.0, 285.0, 52.0, 56.0],
                vec![229.0, 319.0, 75.0, 178.0],
            ],
            vec![
                vec![1467.0, 25.0, 71.0, 153.0],
                vec![1649.0, 108.0, 64.0, 160.0],
                vec![840.0, 227.0, 53.0, 176.0],
                vec![902.0, 71.0, 59.0, 150.0],
                vec![1606.0, 654.0, 122.0, 219.0],
                vec![826.0, 421.0, 102.0, 230.0],
                vec![1143.0, 0.0, 43.0, 82.0],
                vec![1001.0, 0.0, 52.0, 118.0],
                vec![749.0, 214.0, 68.0, 184.0],
                vec![1394.0, 0.0, 51.0, 110.0],
                vec![1323.0, 0.0, 36.0, 46.0],
                vec![226.0, 319.0, 70.0, 183.0],
                vec![686.0, 957.0, 129.0, 123.0],
                vec![174.0, 852.0, 112.0, 228.0],
                vec![1818.0, 170.0, 69.0, 140.0],
                vec![329.0, 225.0, 70.0, 181.0],
                vec![1251.0, 0.0, 28.0, 10.0],
                vec![1824.0, 285.0, 53.0, 56.0],
            ],
            vec![
                vec![1466.0, 22.0, 74.0, 155.0],
                vec![1649.0, 107.0, 64.0, 161.0],
                vec![902.0, 71.0, 63.0, 150.0],
                vec![842.0, 226.0, 52.0, 173.0],
                vec![1143.0, 0.0, 41.0, 82.0],
                vec![1003.0, 0.0, 52.0, 117.0],
                vec![160.0, 855.0, 115.0, 225.0],
                vec![820.0, 424.0, 107.0, 222.0],
                vec![1610.0, 652.0, 120.0, 271.0],
                vec![1324.0, 0.0, 36.0, 45.0],
                vec![753.0, 210.0, 66.0, 179.0],
                vec![1399.0, 0.0, 51.0, 108.0],
                vec![333.0, 223.0, 71.0, 181.0],
                vec![1826.0, 172.0, 60.0, 136.0],
                vec![673.0, 949.0, 150.0, 130.0],
                vec![223.0, 322.0, 68.0, 182.0],
                vec![1823.0, 287.0, 54.0, 54.0],
                vec![1255.0, 0.0, 25.0, 10.0],
            ],
            vec![
                vec![1466.0, 22.0, 73.0, 154.0],
                vec![1649.0, 107.0, 64.0, 161.0],
                vec![904.0, 69.0, 63.0, 152.0],
                vec![845.0, 225.0, 53.0, 177.0],
                vec![1141.0, 0.0, 42.0, 82.0],
                vec![1005.0, 0.0, 52.0, 117.0],
                vec![1609.0, 663.0, 121.0, 210.0],
                vec![756.0, 207.0, 65.0, 178.0],
                vec![673.0, 943.0, 156.0, 135.0],
                vec![338.0, 224.0, 72.0, 180.0],
                vec![157.0, 859.0, 117.0, 221.0],
                vec![814.0, 425.0, 113.0, 219.0],
                vec![1324.0, 0.0, 37.0, 44.0],
                vec![1402.0, 0.0, 52.0, 96.0],
                vec![1829.0, 169.0, 58.0, 144.0],
                vec![1824.0, 289.0, 53.0, 51.0],
                vec![1256.0, 0.0, 24.0, 10.0],
                vec![217.0, 320.0, 70.0, 185.0],
                vec![1770.0, 607.0, 66.0, 155.0],
            ],
            vec![
                vec![1464.0, 21.0, 76.0, 154.0],
                vec![1609.0, 661.0, 124.0, 243.0],
                vec![906.0, 69.0, 64.0, 150.0],
                vec![1649.0, 106.0, 64.0, 162.0],
                vec![1141.0, 0.0, 40.0, 82.0],
                vec![848.0, 225.0, 53.0, 178.0],
                vec![1006.0, 0.0, 53.0, 116.0],
                vec![809.0, 426.0, 122.0, 216.0],
                vec![760.0, 206.0, 64.0, 178.0],
                vec![674.0, 938.0, 156.0, 141.0],
                vec![151.0, 865.0, 128.0, 215.0],
                vec![340.0, 222.0, 73.0, 185.0],
                vec![1326.0, 0.0, 36.0, 43.0],
                vec![1835.0, 165.0, 55.0, 152.0],
                vec![1406.0, 0.0, 52.0, 94.0],
                vec![214.0, 324.0, 75.0, 190.0],
                vec![1824.0, 291.0, 52.0, 49.0],
                vec![1257.0, 0.0, 26.0, 10.0],
            ],
            vec![
                vec![1611.0, 663.0, 122.0, 248.0],
                vec![1463.0, 22.0, 79.0, 150.0],
                vec![1141.0, 0.0, 39.0, 83.0],
                vec![908.0, 68.0, 68.0, 149.0],
                vec![1649.0, 105.0, 63.0, 164.0],
                vec![1007.0, 0.0, 53.0, 115.0],
                vec![851.0, 222.0, 51.0, 181.0],
                vec![679.0, 931.0, 153.0, 148.0],
                vec![806.0, 429.0, 111.0, 211.0],
                vec![1841.0, 164.0, 48.0, 155.0],
                vec![1410.0, 0.0, 51.0, 92.0],
                vec![141.0, 869.0, 127.0, 211.0],
                vec![762.0, 203.0, 64.0, 181.0],
                vec![343.0, 221.0, 73.0, 184.0],
                vec![1326.0, 0.0, 37.0, 42.0],
                vec![203.0, 334.0, 83.0, 188.0],
                vec![1825.0, 294.0, 52.0, 46.0],
            ],
            vec![
                vec![1611.0, 663.0, 122.0, 248.0],
                vec![1463.0, 22.0, 79.0, 150.0],
                vec![1141.0, 0.0, 39.0, 84.0],
                vec![908.0, 68.0, 68.0, 149.0],
                vec![1649.0, 105.0, 63.0, 164.0],
                vec![1008.0, 0.0, 52.0, 115.0],
                vec![851.0, 222.0, 51.0, 181.0],
                vec![678.0, 931.0, 154.0, 148.0],
                vec![806.0, 428.0, 110.0, 213.0],
                vec![1841.0, 163.0, 48.0, 156.0],
                vec![1410.0, 0.0, 51.0, 92.0],
                vec![762.0, 203.0, 64.0, 182.0],
                vec![141.0, 869.0, 128.0, 211.0],
                vec![343.0, 221.0, 73.0, 185.0],
                vec![1326.0, 0.0, 37.0, 42.0],
                vec![203.0, 338.0, 83.0, 184.0],
                vec![1825.0, 293.0, 52.0, 47.0],
            ],
            vec![
                vec![1464.0, 23.0, 79.0, 147.0],
                vec![1611.0, 663.0, 122.0, 259.0],
                vec![1140.0, 0.0, 40.0, 86.0],
                vec![853.0, 221.0, 52.0, 181.0],
                vec![914.0, 67.0, 67.0, 147.0],
                vec![1009.0, 0.0, 52.0, 114.0],
                vec![689.0, 923.0, 143.0, 157.0],
                vec![1646.0, 103.0, 63.0, 160.0],
                vec![766.0, 204.0, 62.0, 178.0],
                vec![805.0, 434.0, 108.0, 226.0],
                vec![1413.0, 0.0, 50.0, 90.0],
                vec![349.0, 221.0, 67.0, 180.0],
                vec![1845.0, 164.0, 44.0, 156.0],
                vec![136.0, 879.0, 128.0, 201.0],
                vec![202.0, 346.0, 82.0, 175.0],
                vec![1326.0, 0.0, 37.0, 42.0],
                vec![1769.0, 610.0, 64.0, 161.0],
            ],
            vec![
                vec![1613.0, 668.0, 122.0, 264.0],
                vec![1463.0, 22.0, 82.0, 147.0],
                vec![854.0, 221.0, 53.0, 177.0],
                vec![769.0, 200.0, 63.0, 181.0],
                vec![1138.0, 0.0, 42.0, 88.0],
                vec![1646.0, 103.0, 64.0, 160.0],
                vec![917.0, 66.0, 67.0, 146.0],
                vec![802.0, 438.0, 109.0, 237.0],
                vec![1011.0, 0.0, 52.0, 114.0],
                vec![696.0, 920.0, 139.0, 160.0],
                vec![1417.0, 0.0, 50.0, 86.0],
                vec![129.0, 887.0, 123.0, 193.0],
                vec![1848.0, 164.0, 41.0, 159.0],
                vec![195.0, 331.0, 82.0, 202.0],
                vec![353.0, 219.0, 64.0, 179.0],
                vec![1330.0, 0.0, 29.0, 42.0],
                vec![1734.0, 0.0, 32.0, 16.0],
            ],
            vec![
                vec![1463.0, 22.0, 82.0, 148.0],
                vec![1614.0, 669.0, 123.0, 273.0],
                vec![797.0, 443.0, 114.0, 235.0],
                vec![1646.0, 102.0, 63.0, 161.0],
                vec![772.0, 198.0, 68.0, 184.0],
                vec![856.0, 220.0, 54.0, 175.0],
                vec![920.0, 66.0, 66.0, 147.0],
                vec![1135.0, 0.0, 45.0, 89.0],
                vec![1011.0, 0.0, 53.0, 114.0],
                vec![697.0, 917.0, 143.0, 162.0],
                vec![1419.0, 0.0, 50.0, 84.0],
                vec![189.0, 338.0, 81.0, 189.0],
                vec![123.0, 894.0, 120.0, 186.0],
                vec![1331.0, 0.0, 27.0, 41.0],
            ],
            vec![
                vec![1462.0, 21.0, 84.0, 148.0],
                vec![1646.0, 101.0, 62.0, 159.0],
                vec![793.0, 446.0, 115.0, 236.0],
                vec![1615.0, 682.0, 121.0, 268.0],
                vec![775.0, 203.0, 72.0, 178.0],
                vec![922.0, 65.0, 68.0, 149.0],
                vec![1134.0, 0.0, 48.0, 89.0],
                vec![1014.0, 0.0, 52.0, 114.0],
                vec![859.0, 218.0, 54.0, 173.0],
                vec![113.0, 901.0, 119.0, 179.0],
                vec![703.0, 915.0, 140.0, 165.0],
                vec![1423.0, 0.0, 51.0, 85.0],
                vec![186.0, 336.0, 71.0, 195.0],
                vec![1331.0, 0.0, 29.0, 41.0],
                vec![358.0, 208.0, 67.0, 180.0],
                vec![1721.0, 0.0, 46.0, 19.0],
            ],
            vec![
                vec![1462.0, 19.0, 84.0, 149.0],
                vec![1646.0, 101.0, 62.0, 155.0],
                vec![1616.0, 681.0, 122.0, 276.0],
                vec![788.0, 450.0, 117.0, 234.0],
                vec![776.0, 203.0, 72.0, 177.0],
                vec![922.0, 65.0, 69.0, 148.0],
                vec![1015.0, 0.0, 51.0, 113.0],
                vec![862.0, 216.0, 54.0, 169.0],
                vec![1132.0, 0.0, 49.0, 90.0],
                vec![106.0, 910.0, 116.0, 169.0],
                vec![1426.0, 0.0, 51.0, 82.0],
                vec![701.0, 909.0, 149.0, 169.0],
                vec![180.0, 336.0, 68.0, 196.0],
                vec![1332.0, 0.0, 28.0, 41.0],
                vec![1719.0, 0.0, 48.0, 19.0],
            ],
            vec![
                vec![1462.0, 19.0, 84.0, 149.0],
                vec![1646.0, 101.0, 62.0, 155.0],
                vec![1616.0, 682.0, 122.0, 274.0],
                vec![788.0, 450.0, 117.0, 233.0],
                vec![776.0, 203.0, 72.0, 178.0],
                vec![922.0, 65.0, 69.0, 148.0],
                vec![862.0, 216.0, 54.0, 169.0],
                vec![1015.0, 0.0, 51.0, 113.0],
                vec![1132.0, 0.0, 49.0, 90.0],
                vec![1426.0, 0.0, 51.0, 82.0],
                vec![106.0, 910.0, 116.0, 169.0],
                vec![700.0, 909.0, 150.0, 169.0],
                vec![180.0, 336.0, 68.0, 196.0],
                vec![1332.0, 0.0, 28.0, 41.0],
                vec![1719.0, 0.0, 48.0, 19.0],
            ],
            vec![
                vec![1645.0, 99.0, 63.0, 155.0],
                vec![1463.0, 17.0, 84.0, 147.0],
                vec![1616.0, 689.0, 122.0, 285.0],
                vec![864.0, 213.0, 54.0, 172.0],
                vec![777.0, 201.0, 71.0, 181.0],
                vec![1015.0, 0.0, 53.0, 114.0],
                vec![1132.0, 0.0, 48.0, 91.0],
                vec![92.0, 915.0, 121.0, 165.0],
                vec![925.0, 65.0, 66.0, 146.0],
                vec![783.0, 451.0, 123.0, 238.0],
                vec![713.0, 902.0, 142.0, 178.0],
                vec![1430.0, 0.0, 51.0, 84.0],
                vec![170.0, 339.0, 72.0, 188.0],
                vec![1718.0, 0.0, 50.0, 20.0],
                vec![359.0, 206.0, 71.0, 179.0],
                vec![1329.0, 0.0, 34.0, 41.0],
            ],
            vec![
                vec![1645.0, 98.0, 64.0, 154.0],
                vec![1465.0, 15.0, 82.0, 148.0],
                vec![1616.0, 686.0, 121.0, 292.0],
                vec![867.0, 210.0, 53.0, 173.0],
                vec![779.0, 201.0, 72.0, 175.0],
                vec![930.0, 64.0, 64.0, 146.0],
                vec![1015.0, 0.0, 54.0, 114.0],
                vec![1131.0, 0.0, 48.0, 92.0],
                vec![721.0, 900.0, 137.0, 180.0],
                vec![84.0, 919.0, 125.0, 161.0],
                vec![1433.0, 0.0, 52.0, 84.0],
                vec![781.0, 449.0, 124.0, 237.0],
                vec![168.0, 340.0, 69.0, 190.0],
                vec![1717.0, 0.0, 51.0, 20.0],
                vec![359.0, 207.0, 76.0, 177.0],
                vec![1325.0, 0.0, 41.0, 42.0],
            ],
            vec![
                vec![1615.0, 694.0, 124.0, 280.0],
                vec![1646.0, 97.0, 63.0, 154.0],
                vec![1468.0, 13.0, 81.0, 149.0],
                vec![868.0, 210.0, 55.0, 172.0],
                vec![783.0, 199.0, 70.0, 176.0],
                vec![1130.0, 0.0, 48.0, 92.0],
                vec![930.0, 65.0, 66.0, 146.0],
                vec![1015.0, 0.0, 56.0, 115.0],
                vec![78.0, 922.0, 116.0, 158.0],
                vec![722.0, 892.0, 141.0, 188.0],
                vec![777.0, 457.0, 112.0, 241.0],
                vec![362.0, 207.0, 84.0, 175.0],
                vec![160.0, 350.0, 74.0, 185.0],
                vec![1436.0, 0.0, 51.0, 86.0],
                vec![1718.0, 0.0, 49.0, 21.0],
            ],
            vec![
                vec![1645.0, 96.0, 65.0, 154.0],
                vec![1613.0, 696.0, 130.0, 278.0],
                vec![1474.0, 12.0, 76.0, 149.0],
                vec![790.0, 194.0, 67.0, 178.0],
                vec![1129.0, 0.0, 48.0, 93.0],
                vec![931.0, 66.0, 65.0, 146.0],
                vec![870.0, 208.0, 57.0, 172.0],
                vec![1016.0, 0.0, 57.0, 114.0],
                vec![364.0, 207.0, 95.0, 175.0],
                vec![723.0, 890.0, 145.0, 190.0],
                vec![69.0, 926.0, 115.0, 154.0],
                vec![774.0, 458.0, 107.0, 240.0],
                vec![1439.0, 0.0, 53.0, 88.0],
                vec![155.0, 348.0, 79.0, 189.0],
                vec![1718.0, 0.0, 48.0, 21.0],
            ],
            vec![
                vec![1645.0, 95.0, 65.0, 154.0],
                vec![1612.0, 699.0, 132.0, 276.0],
                vec![1475.0, 11.0, 76.0, 150.0],
                vec![874.0, 208.0, 62.0, 173.0],
                vec![796.0, 191.0, 64.0, 176.0],
                vec![1129.0, 0.0, 46.0, 93.0],
                vec![932.0, 68.0, 65.0, 144.0],
                vec![729.0, 884.0, 144.0, 196.0],
                vec![1018.0, 0.0, 55.0, 113.0],
                vec![54.0, 929.0, 119.0, 151.0],
                vec![363.0, 207.0, 103.0, 175.0],
                vec![139.0, 350.0, 96.0, 193.0],
                vec![1719.0, 0.0, 45.0, 21.0],
                vec![769.0, 456.0, 110.0, 199.0],
                vec![1445.0, 0.0, 52.0, 87.0],
            ],
            vec![
                vec![1645.0, 95.0, 65.0, 154.0],
                vec![1612.0, 699.0, 132.0, 275.0],
                vec![1475.0, 11.0, 76.0, 150.0],
                vec![874.0, 208.0, 62.0, 173.0],
                vec![796.0, 191.0, 64.0, 177.0],
                vec![1129.0, 0.0, 46.0, 93.0],
                vec![932.0, 68.0, 66.0, 144.0],
                vec![730.0, 884.0, 142.0, 196.0],
                vec![1018.0, 0.0, 55.0, 113.0],
                vec![54.0, 929.0, 118.0, 151.0],
                vec![363.0, 207.0, 103.0, 175.0],
                vec![140.0, 350.0, 95.0, 192.0],
                vec![1719.0, 0.0, 45.0, 21.0],
                vec![769.0, 456.0, 110.0, 198.0],
                vec![1446.0, 0.0, 51.0, 87.0],
            ],
            vec![
                vec![1645.0, 95.0, 65.0, 155.0],
                vec![1477.0, 11.0, 76.0, 149.0],
                vec![800.0, 190.0, 63.0, 173.0],
                vec![876.0, 207.0, 65.0, 172.0],
                vec![1614.0, 706.0, 128.0, 270.0],
                vec![936.0, 66.0, 63.0, 144.0],
                vec![1129.0, 0.0, 45.0, 94.0],
                vec![382.0, 207.0, 85.0, 176.0],
                vec![1018.0, 0.0, 56.0, 107.0],
                vec![732.0, 880.0, 143.0, 200.0],
                vec![50.0, 934.0, 116.0, 146.0],
                vec![136.0, 352.0, 100.0, 192.0],
                vec![765.0, 459.0, 113.0, 192.0],
                vec![1448.0, 0.0, 51.0, 88.0],
                vec![1743.0, 654.0, 77.0, 167.0],
                vec![1717.0, 0.0, 38.0, 21.0],
            ],
            vec![
                vec![1645.0, 96.0, 65.0, 154.0],
                vec![1477.0, 11.0, 75.0, 149.0],
                vec![803.0, 190.0, 62.0, 173.0],
                vec![877.0, 206.0, 64.0, 173.0],
                vec![1620.0, 708.0, 126.0, 264.0],
                vec![381.0, 207.0, 86.0, 176.0],
                vec![1128.0, 0.0, 43.0, 94.0],
                vec![940.0, 64.0, 59.0, 145.0],
                vec![1018.0, 0.0, 58.0, 106.0],
                vec![741.0, 873.0, 135.0, 205.0],
                vec![758.0, 460.0, 115.0, 209.0],
                vec![134.0, 351.0, 99.0, 197.0],
                vec![35.0, 941.0, 126.0, 139.0],
                vec![1451.0, 0.0, 52.0, 88.0],
                vec![1717.0, 0.0, 39.0, 22.0],
                vec![1741.0, 657.0, 75.0, 179.0],
            ],
            vec![
                vec![1645.0, 96.0, 65.0, 152.0],
                vec![1478.0, 11.0, 78.0, 149.0],
                vec![1621.0, 715.0, 124.0, 263.0],
                vec![806.0, 188.0, 62.0, 174.0],
                vec![879.0, 205.0, 62.0, 173.0],
                vec![942.0, 59.0, 59.0, 144.0],
                vec![758.0, 465.0, 108.0, 230.0],
                vec![1127.0, 0.0, 41.0, 95.0],
                vec![383.0, 207.0, 85.0, 175.0],
                vec![745.0, 865.0, 138.0, 215.0],
                vec![1019.0, 0.0, 56.0, 105.0],
                vec![31.0, 952.0, 114.0, 128.0],
                vec![124.0, 354.0, 103.0, 203.0],
                vec![1718.0, 0.0, 35.0, 21.0],
                vec![1738.0, 659.0, 80.0, 179.0],
                vec![1456.0, 0.0, 51.0, 58.0],
            ],
            vec![
                vec![1626.0, 723.0, 119.0, 250.0],
                vec![1646.0, 96.0, 64.0, 151.0],
                vec![1481.0, 12.0, 75.0, 146.0],
                vec![753.0, 465.0, 108.0, 248.0],
                vec![881.0, 203.0, 63.0, 174.0],
                vec![942.0, 57.0, 58.0, 149.0],
                vec![749.0, 866.0, 138.0, 214.0],
                vec![810.0, 187.0, 62.0, 178.0],
                vec![387.0, 207.0, 82.0, 174.0],
                vec![1126.0, 0.0, 40.0, 95.0],
                vec![22.0, 960.0, 117.0, 120.0],
                vec![1027.0, 0.0, 48.0, 105.0],
                vec![116.0, 361.0, 105.0, 193.0],
                vec![1736.0, 664.0, 79.0, 175.0],
                vec![1459.0, 0.0, 48.0, 56.0],
                vec![1719.0, 0.0, 28.0, 21.0],
            ],
            vec![
                vec![1630.0, 726.0, 118.0, 255.0],
                vec![1481.0, 11.0, 76.0, 150.0],
                vec![744.0, 466.0, 117.0, 255.0],
                vec![810.0, 186.0, 65.0, 177.0],
                vec![750.0, 864.0, 141.0, 216.0],
                vec![1647.0, 94.0, 63.0, 153.0],
                vec![945.0, 58.0, 58.0, 148.0],
                vec![889.0, 203.0, 59.0, 169.0],
                vec![1121.0, 0.0, 43.0, 97.0],
                vec![392.0, 204.0, 77.0, 175.0],
                vec![1028.0, 0.0, 48.0, 105.0],
                vec![8.0, 971.0, 129.0, 109.0],
                vec![113.0, 363.0, 96.0, 192.0],
                vec![1462.0, 0.0, 48.0, 79.0],
            ],
            vec![
                vec![1630.0, 726.0, 118.0, 255.0],
                vec![1481.0, 11.0, 76.0, 150.0],
                vec![810.0, 186.0, 65.0, 178.0],
                vec![744.0, 466.0, 117.0, 255.0],
                vec![750.0, 864.0, 142.0, 216.0],
                vec![1647.0, 94.0, 63.0, 153.0],
                vec![945.0, 58.0, 58.0, 148.0],
                vec![889.0, 203.0, 59.0, 169.0],
                vec![1121.0, 0.0, 43.0, 97.0],
                vec![392.0, 204.0, 77.0, 175.0],
                vec![1028.0, 0.0, 48.0, 105.0],
                vec![7.0, 971.0, 132.0, 108.0],
                vec![113.0, 362.0, 97.0, 193.0],
                vec![1462.0, 0.0, 48.0, 80.0],
            ],
            vec![
                vec![1632.0, 736.0, 118.0, 235.0],
                vec![736.0, 474.0, 119.0, 246.0],
                vec![894.0, 200.0, 55.0, 168.0],
                vec![1650.0, 91.0, 62.0, 150.0],
                vec![1483.0, 17.0, 72.0, 134.0],
                vec![756.0, 857.0, 141.0, 223.0],
                vec![812.0, 187.0, 67.0, 176.0],
                vec![947.0, 60.0, 57.0, 143.0],
                vec![1120.0, 0.0, 43.0, 98.0],
                vec![1029.0, 0.0, 48.0, 104.0],
                vec![397.0, 199.0, 75.0, 177.0],
                vec![1.0, 979.0, 126.0, 101.0],
                vec![105.0, 366.0, 78.0, 188.0],
                vec![1731.0, 675.0, 77.0, 172.0],
                vec![1466.0, 1.0, 49.0, 76.0],
                vec![1335.0, 0.0, 35.0, 31.0],
            ],
            vec![
                vec![900.0, 198.0, 50.0, 170.0],
                vec![1635.0, 741.0, 119.0, 229.0],
                vec![1650.0, 89.0, 62.0, 150.0],
                vec![1485.0, 9.0, 73.0, 145.0],
                vec![732.0, 482.0, 119.0, 236.0],
                vec![813.0, 186.0, 69.0, 176.0],
                vec![760.0, 854.0, 139.0, 226.0],
                vec![949.0, 59.0, 57.0, 145.0],
                vec![1119.0, 0.0, 44.0, 100.0],
                vec![1029.0, 0.0, 51.0, 104.0],
                vec![403.0, 196.0, 70.0, 174.0],
                vec![0.0, 988.0, 110.0, 92.0],
                vec![1728.0, 677.0, 79.0, 169.0],
                vec![102.0, 364.0, 76.0, 187.0],
                vec![1335.0, 0.0, 34.0, 31.0],
                vec![1470.0, 0.0, 48.0, 77.0],
            ],
            vec![
                vec![901.0, 196.0, 52.0, 169.0],
                vec![760.0, 851.0, 142.0, 228.0],
                vec![1485.0, 8.0, 73.0, 142.0],
                vec![1650.0, 87.0, 63.0, 151.0],
                vec![1638.0, 747.0, 117.0, 233.0],
                vec![816.0, 184.0, 69.0, 179.0],
                vec![727.0, 486.0, 120.0, 234.0],
                vec![951.0, 59.0, 56.0, 144.0],
                vec![409.0, 193.0, 65.0, 178.0],
                vec![1030.0, 0.0, 51.0, 104.0],
                vec![1116.0, 0.0, 44.0, 99.0],
                vec![0.0, 995.0, 102.0, 85.0],
                vec![98.0, 366.0, 69.0, 193.0],
                vec![1726.0, 682.0, 76.0, 137.0],
                vec![1472.0, 1.0, 47.0, 74.0],
            ],
            vec![
                vec![1650.0, 84.0, 63.0, 153.0],
                vec![1485.0, 6.0, 73.0, 141.0],
                vec![1640.0, 751.0, 117.0, 227.0],
                vec![818.0, 183.0, 71.0, 180.0],
                vec![905.0, 194.0, 52.0, 170.0],
                vec![759.0, 844.0, 147.0, 235.0],
                vec![953.0, 59.0, 60.0, 144.0],
                vec![411.0, 193.0, 66.0, 177.0],
                vec![728.0, 484.0, 119.0, 240.0],
                vec![1114.0, 0.0, 45.0, 100.0],
                vec![1031.0, 0.0, 53.0, 103.0],
                vec![0.0, 1002.0, 93.0, 78.0],
                vec![93.0, 368.0, 72.0, 192.0],
                vec![1725.0, 687.0, 74.0, 133.0],
            ],
            vec![
                vec![1650.0, 83.0, 64.0, 152.0],
                vec![1485.0, 3.0, 74.0, 144.0],
                vec![908.0, 193.0, 51.0, 171.0],
                vec![1643.0, 753.0, 114.0, 226.0],
                vec![955.0, 57.0, 63.0, 145.0],
                vec![821.0, 183.0, 72.0, 177.0],
                vec![1112.0, 0.0, 46.0, 100.0],
                vec![765.0, 837.0, 141.0, 242.0],
                vec![724.0, 482.0, 116.0, 239.0],
                vec![414.0, 191.0, 65.0, 179.0],
                vec![1033.0, 0.0, 52.0, 103.0],
                vec![0.0, 1006.0, 82.0, 74.0],
                vec![87.0, 373.0, 77.0, 185.0],
                vec![1726.0, 689.0, 69.0, 130.0],
                vec![1706.0, 0.0, 30.0, 25.0],
                vec![1480.0, 0.0, 46.0, 40.0],
            ],
            vec![
                vec![1650.0, 83.0, 64.0, 152.0],
                vec![1484.0, 3.0, 76.0, 144.0],
                vec![908.0, 193.0, 51.0, 171.0],
                vec![1642.0, 753.0, 115.0, 226.0],
                vec![955.0, 57.0, 63.0, 145.0],
                vec![821.0, 182.0, 72.0, 178.0],
                vec![1112.0, 0.0, 46.0, 100.0],
                vec![765.0, 837.0, 142.0, 242.0],
                vec![414.0, 191.0, 65.0, 178.0],
                vec![724.0, 482.0, 116.0, 240.0],
                vec![1033.0, 0.0, 52.0, 103.0],
                vec![87.0, 373.0, 77.0, 185.0],
                vec![0.0, 1006.0, 81.0, 74.0],
                vec![1726.0, 689.0, 69.0, 131.0],
                vec![1706.0, 0.0, 29.0, 24.0],
                vec![1480.0, 0.0, 45.0, 41.0],
            ],
            vec![
                vec![1483.0, 2.0, 79.0, 145.0],
                vec![1650.0, 82.0, 64.0, 154.0],
                vec![911.0, 193.0, 49.0, 172.0],
                vec![776.0, 825.0, 135.0, 255.0],
                vec![1644.0, 758.0, 117.0, 226.0],
                vec![824.0, 181.0, 71.0, 173.0],
                vec![957.0, 52.0, 63.0, 153.0],
                vec![420.0, 190.0, 64.0, 177.0],
                vec![1112.0, 0.0, 44.0, 101.0],
                vec![718.0, 489.0, 111.0, 247.0],
                vec![1034.0, 0.0, 52.0, 103.0],
                vec![1726.0, 690.0, 67.0, 131.0],
                vec![1703.0, 0.0, 32.0, 25.0],
                vec![79.0, 369.0, 87.0, 198.0],
                vec![1336.0, 0.0, 33.0, 31.0],
                vec![0.0, 1009.0, 82.0, 71.0],
            ],
            vec![
                vec![1482.0, 1.0, 79.0, 146.0],
                vec![1650.0, 82.0, 64.0, 153.0],
                vec![830.0, 174.0, 68.0, 173.0],
                vec![913.0, 192.0, 49.0, 173.0],
                vec![1645.0, 762.0, 118.0, 247.0],
                vec![788.0, 819.0, 127.0, 261.0],
                vec![420.0, 189.0, 69.0, 176.0],
                vec![713.0, 495.0, 116.0, 238.0],
                vec![1111.0, 0.0, 42.0, 101.0],
                vec![957.0, 54.0, 63.0, 153.0],
                vec![1036.0, 0.0, 51.0, 100.0],
                vec![1723.0, 692.0, 71.0, 128.0],
                vec![1702.0, 0.0, 33.0, 24.0],
                vec![70.0, 375.0, 93.0, 206.0],
                vec![1336.0, 0.0, 33.0, 31.0],
            ],
            vec![
                vec![1650.0, 82.0, 65.0, 153.0],
                vec![1481.0, 0.0, 79.0, 147.0],
                vec![786.0, 815.0, 131.0, 265.0],
                vec![834.0, 174.0, 65.0, 169.0],
                vec![914.0, 192.0, 51.0, 171.0],
                vec![1642.0, 764.0, 119.0, 260.0],
                vec![422.0, 191.0, 69.0, 174.0],
                vec![957.0, 57.0, 64.0, 148.0],
                vec![710.0, 494.0, 116.0, 243.0],
                vec![1111.0, 0.0, 40.0, 101.0],
                vec![1037.0, 0.0, 51.0, 100.0],
                vec![1700.0, 0.0, 34.0, 23.0],
                vec![1718.0, 697.0, 77.0, 129.0],
                vec![1337.0, 0.0, 32.0, 30.0],
                vec![1488.0, 0.0, 48.0, 31.0],
            ],
            vec![
                vec![1650.0, 82.0, 65.0, 152.0],
                vec![915.0, 192.0, 51.0, 170.0],
                vec![791.0, 811.0, 134.0, 269.0],
                vec![1480.0, 0.0, 81.0, 147.0],
                vec![840.0, 172.0, 62.0, 172.0],
                vec![1644.0, 762.0, 118.0, 279.0],
                vec![961.0, 56.0, 63.0, 147.0],
                vec![426.0, 192.0, 67.0, 171.0],
                vec![708.0, 496.0, 108.0, 242.0],
                vec![1109.0, 0.0, 41.0, 103.0],
                vec![1038.0, 0.0, 52.0, 100.0],
                vec![1701.0, 0.0, 33.0, 24.0],
                vec![1716.0, 699.0, 76.0, 126.0],
                vec![61.0, 382.0, 95.0, 207.0],
                vec![1339.0, 0.0, 31.0, 29.0],
            ],
        ];

        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, None, None, None);
        let mut tracker = Tracker::new(metric, None, None, None);

        detections.truncate(50);
        detections
            .iter()
            .enumerate()
            .for_each(|(iteration, detection)| {
                println!("\n\n{}", iteration);

                &tracker.predict();
                &tracker.update(
                    &detection
                        .iter()
                        .map(|d| {
                            Detection::new(
                                BoundingBox::new(d[0], d[1], d[2], d[3]),
                                1.0,
                                None,
                                None,
                            )
                        })
                        .collect::<Vec<Detection>>(),
                );

                // for debugging
                for track in &tracker.tracks {
                    println!(
                        "{}: {:?} {:?} {:?} {:?} {:?}",
                        iteration,
                        track.track_id(),
                        track.state(),
                        track.bbox().to_tlwh(),
                        tracker
                            .metric
                            .track_features(*track.track_id())
                            .unwrap_or(&Array2::<f32>::zeros((0, *tracker.metric.feature_length())))
                            .nrows(),
                        track.match_source(),
                    );
                }
            });
    }
}
