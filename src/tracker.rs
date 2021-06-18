use crate::{Detection, KalmanFilter, NearestNeighborDistanceMetric, Track, Match};

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
    pub fn update(&mut self, detections: Vec<Detection>) {

        // Run matching cascade.
        let (matches, unmatched_tracks, unmatched_detections) = self.match_impl(&detections);

        // Update track set.
        for m in matches {
            self.tracks.get_mut(m.track_idx).unwrap().update(&self.kf, detections.get(m.detection_idx).unwrap());
        }
        for unmatched_track in unmatched_tracks {
            self.tracks.get_mut(unmatched_track).unwrap().mark_missed();
        }
        for detection_idx in unmatched_detections {
            self.initiate_track(detections.get(detection_idx).unwrap().to_owned());
        }

        // Update distance metric.
        let active_targets: Vec<usize> = self.tracks.iter().filter(|track| track.is_confirmed()).map(|track| track.track_id).collect();

        self.tracks.iter_mut().filter(|track| track.is_confirmed())

    }

    fn match_impl(&self, detections: &Vec<Detection>) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
        // fn gated_metric(tracks: Vec<Track>) -> Array2<f32> {
        //     arr2::<f32, _>(&[[]])
        // }

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

        // # Associate confirmed tracks using appearance features.
        // matches_a, unmatched_tracks_a, unmatched_detections = \
        //     linear_assignment.matching_cascade(
        //         gated_metric, self.metric.matching_threshold, self.max_age,
        //         self.tracks, detections, confirmed_tracks)
        let matches_a: Vec<Match> = vec![];
        let unmatched_tracks_a: Vec<usize> = vec![];
        let unmatched_detections: Vec<usize> = vec![];

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        let iou_track_candidates = [
            unconfirmed_tracks,
            unmatched_tracks_a
                .iter()
                .filter(|k| self.tracks.get(**k).unwrap().time_since_update == 0)
                .map(|v| v.to_owned())
                .collect::<Vec<usize>>(),
        ]
        .concat();

        let unmatched_tracks_a = unmatched_tracks_a
            .iter()
            .filter(|k| self.tracks.get(**k).unwrap().time_since_update != 0)
            .map(|v| v.to_owned())
            .collect::<Vec<usize>>();

        // matches_b, unmatched_tracks_b, unmatched_detections = \
        // linear_assignment.min_cost_matching(
        //     iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        //     detections, iou_track_candidates, unmatched_detections)
        let matches_b: Vec<Match> = vec![];
        let unmatched_tracks_b: Vec<usize> = vec![];
        let unmatched_detections: Vec<usize> = vec![];

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
            Some(stack![Axis(0), detection.feature]),
        ));
        self.next_id += 1;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn a() {}
}
