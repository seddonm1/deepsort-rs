use std::rc::Rc;

use crate::*;

use ndarray::*;

/**
This is the multi-target tracker.

Parameters
----------
metric : nn_matching.NearestNeighborDistanceMetric
    A distance metric for measurement-to-track association.
max_age : int
    Maximum number of missed misses before a track is deleted.
n_init : int
    Number of consecutive detections before the track is confirmed. The
    track state is set to `Deleted` if a miss occurs within the first
    `n_init` frames.

Attributes
----------
metric : nn_matching.NearestNeighborDistanceMetric
    The distance metric used for measurement to track association.
max_age : int
    Maximum number of missed misses before a track is deleted.
n_init : int
    Number of frames that a track remains in initialization phase.
kf : kalman_filter.KalmanFilter
    A Kalman filter to filter target trajectories in image space.
tracks : List[Track]
    The list of active tracks at the current time step.
*/
#[derive(Debug)]
pub struct Tracker {
    metric: NearestNeighborDistanceMetric,
    max_iou_distance: f32,
    max_age: usize,
    n_init: usize,
    kf: KalmanFilter,
    tracks: Vec<Track>,
    next_id: usize,
}

impl Tracker {
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

    /**
    Propagate track state distributions one time step forward.
    This function should be called once every time step, before `update`.
    */
    pub fn predict(&mut self) {
        let kf = self.kf.clone();
        self.tracks.iter_mut().for_each(|track| track.predict(&kf));
    }

    /**
    Perform measurement update and track management.

    Parameters
    ----------
    detections : List[deep_sort.detection.Detection]
        A list of detections at the current time step.
    */
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

        let mut features = Array2::<f32>::zeros((0, 128));
        let mut targets: Vec<usize> = vec![];
        self.tracks
            .iter_mut()
            .filter(|track| track.is_confirmed())
            .for_each(|track| {
                features = concatenate![Axis(0), features, *track.features()];
                for _ in 0..track.features().nrows() {
                    targets.push(*track.track_id());
                }
                *track.features_mut() = Array2::zeros((0, 128));
            });

        self.metric
            .partial_fit(&features, &ArrayBase::from(targets), &active_targets)
    }

    fn match_impl(&self, detections: &[Detection]) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
        let metric = self.metric.clone();
        let kf = self.kf.clone();

        let gated_metric = Rc::new(
            move |tracks: &[Track],
                  dets: &[Detection],
                  track_indices: Option<Vec<usize>>,
                  detection_indices: Option<Vec<usize>>|
                  -> Array2<f32> {
                let detection_indices = detection_indices.unwrap();
                let track_indices = track_indices.unwrap();

                let mut features = Array2::<f32>::zeros((0, 128));
                detection_indices.iter().for_each(|i| {
                    features
                        .push_row(dets.get(*i).unwrap().feature().view())
                        .unwrap()
                });
                let targets = track_indices
                    .iter()
                    .map(|i| *tracks.get(*i).unwrap().track_id())
                    .collect::<Vec<usize>>();

                let cost_matrix = metric.distance(&features, &targets);

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
                self.metric.matching_threshold(),
                self.max_age,
                &self.tracks,
                detections,
                Some(confirmed_tracks),
                None,
            );

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        let iou_track_candidates = [
            unconfirmed_tracks,
            unmatched_tracks_a
                .iter()
                .filter(|k| *self.tracks.get(**k).unwrap().time_since_update() == 0)
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
        self.tracks.push(Track::new(
            mean,
            covariance,
            self.next_id,
            self.n_init,
            self.max_age,
            Some(stack![Axis(0), *detection.feature()]),
        ));
        self.next_id += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    use rand::prelude::*;
    use rand_pcg::{Lcg64Xsh32, Pcg32};

    /**
    Returns a psuedo-random (deterministic) f32 between -0.5 and +0.5
    */
    fn next_f32(rng: &mut Lcg64Xsh32) -> f32 {
        (rng.next_u32() as f64 / u32::MAX as f64) as f32 - 0.5
    }

    #[test]
    fn tracker() {
        // deterministic generator
        let mut rng = Pcg32::seed_from_u64(0);
        let mut rnd = Vec::<f32>::with_capacity(500);
        for _ in 0..1000 {
            rnd.push(next_f32(&mut rng));
        }
        // println!("{:?}", random);

        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.2, None);
        let mut tracker = Tracker::new(metric, None, None, None);

        for i in 0..101 {
            println!("{} ---------------------------", i);

            // move up to right
            let d0_x = 0.0 + (i as f32) + rnd.pop().unwrap();
            let d0_y = 0.0 + (i as f32) + rnd.pop().unwrap();
            let d0 = Detection::new(
                arr1::<f32>(&[d0_x, d0_y, 10.0, 10.0]),
                1.0,
                Array1::<f32>::from_elem(128, 0.5),
            );

            // move down to left
            let d1_x = 100.0 - (i as f32) + rnd.pop().unwrap();
            let d1_y = 100.0 - (i as f32) + rnd.pop().unwrap();
            let d1 = Detection::new(
                arr1::<f32>(&[d1_x, d1_y, 8.0, 8.0]),
                1.0,
                Array1::<f32>::from_elem(128, -0.5),
            );

            // move down to left
            let d2_x = 0.0 + (i as f32) + rnd.pop().unwrap();
            let d2_y = 100.0 - (i as f32) + rnd.pop().unwrap();
            let d2 = Detection::new(
                arr1::<f32>(&[d2_x, d2_y, 6.0, 6.0]),
                1.0,
                Array1::<f32>::from_elem(128, 0.1),
            );

            &tracker.predict();
            if i >= 10 && i <= 40 {
                &tracker.update(&[d0, d1, d2]);
            } else {
                &tracker.update(&[d0, d1]);
            }
            for track in &tracker.tracks {
                println!(
                    "{}: {:?} {:?} {:?} {:?}",
                    i,
                    track.track_id(),
                    track.state(),
                    track.to_tlwh(),
                    tracker
                        .metric
                        .features(*track.track_id())
                        .unwrap_or(&Array2::<f32>::zeros((0, 128)))
                        .nrows(),
                );
            }
            println!("")
        }
    }
}
