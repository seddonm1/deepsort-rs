use std::collections::HashSet;

use crate::*;

use ndarray::*;
use pathfinding::kuhn_munkres::kuhn_munkres_min;
use pathfinding::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct Match {
    pub track_idx: usize,
    pub detection_idx: usize,
}

impl Match {
    pub fn new(track_idx: usize, detection_idx: usize) -> Match {
        Match {
            track_idx,
            detection_idx,
        }
    }
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.track_idx == other.track_idx && self.detection_idx == other.detection_idx
    }
}

/**
Solve linear assignment problem.

Parameters
----------
distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
    The distance metric is given a list of tracks and detections as well as
    a list of N track indices and M detection indices. The metric should
    return the NxM dimensional cost matrix, where element (i, j) is the
    association cost between the i-th track in the given track indices and
    the j-th detection in the given detection_indices.
max_distance : float
    Gating threshold. Associations with cost larger than this value are
    disregarded.
tracks : List[track.Track]
    A list of predicted tracks at the current time step.
detections : List[detection.Detection]
    A list of detections at the current time step.
track_indices : List[int]
    List of track indices that maps rows in `cost_matrix` to tracks in
    `tracks` (see description above).
detection_indices : List[int]
    List of detection indices that maps columns in `cost_matrix` to
    detections in `detections` (see description above).

Returns
-------
(List[(int, int)], List[int], List[int])
    Returns a tuple with the following three entries:
    * A list of matched track and detection indices.
    * A list of unmatched track indices.
    * A list of unmatched detection indices.
*/
pub fn min_cost_matching(
    distance_metric: fn(
        &[Track],
        &[Detection],
        Option<Vec<usize>>,
        Option<Vec<usize>>,
    ) -> Array2<f32>,
    max_distance: f32,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    if detection_indices.is_empty() || track_indices.is_empty() {
        (vec![], track_indices, detection_indices)
    } else {
        let cost_matrix = (distance_metric)(
            tracks,
            detections,
            Some(track_indices.clone()),
            Some(detection_indices.clone()),
        );

        // multiply by large constant to convert to i64 which satisfies Matrix requirements (Ord)
        let cost_vec = cost_matrix
            .mapv(|v| (v.min(max_distance + 1e-5) * 1_000_000_000.0) as i64)
            .iter()
            .cloned()
            .collect::<Vec<i64>>();

        let matrix = Matrix::from_vec(cost_matrix.nrows(), cost_matrix.ncols(), cost_vec).unwrap();
        let (_, col_indices) = kuhn_munkres_min(&matrix);
        let row_indices = (0..col_indices.len()).collect::<Vec<usize>>();

        let mut matches: Vec<Match> = vec![];
        let mut unmatched_tracks: Vec<usize> = vec![];
        let mut unmatched_detections: Vec<usize> = vec![];

        detection_indices
            .iter()
            .enumerate()
            .for_each(|(col, detection_idx)| {
                if !col_indices.contains(&col) {
                    unmatched_detections.push(*detection_idx);
                };
            });
        track_indices
            .iter()
            .enumerate()
            .for_each(|(row, track_idx)| {
                if !row_indices.contains(&row) {
                    unmatched_tracks.push(*track_idx);
                };
            });
        row_indices
            .iter()
            .zip(col_indices.iter())
            .for_each(|(row, col)| {
                let track_idx = *track_indices.get(*row).unwrap();
                let detection_idx = *detection_indices.get(*col).unwrap();

                if cost_matrix[[*row, *col]] > max_distance {
                    unmatched_tracks.push(track_idx);
                    unmatched_detections.push(detection_idx);
                } else {
                    matches.push(Match::new(track_idx, detection_idx));
                }
            });

        (matches, unmatched_tracks, unmatched_detections)
    }
}

/**
Run matching cascade.

Parameters
----------
distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
    The distance metric is given a list of tracks and detections as well as
    a list of N track indices and M detection indices. The metric should
    return the NxM dimensional cost matrix, where element (i, j) is the
    association cost between the i-th track in the given track indices and
    the j-th detection in the given detection indices.
max_distance : float
    Gating threshold. Associations with cost larger than this value are
    disregarded.
cascade_depth: int
    The cascade depth, should be se to the maximum track age.
tracks : List[track.Track]
    A list of predicted tracks at the current time step.
detections : List[detection.Detection]
    A list of detections at the current time step.
track_indices : Optional[List[int]]
    List of track indices that maps rows in `cost_matrix` to tracks in
    `tracks` (see description above). Defaults to all tracks.
detection_indices : Optional[List[int]]
    List of detection indices that maps columns in `cost_matrix` to
    detections in `detections` (see description above). Defaults to all
    detections.

Returns
-------
(List[(int, int)], List[int], List[int])
    Returns a tuple with the following three entries:
    * A list of matched track and detection indices.
    * A list of unmatched track indices.
    * A list of unmatched detection indices.
*/
pub fn matching_cascade(
    distance_metric: fn(
        &[Track],
        &[Detection],
        Option<Vec<usize>>,
        Option<Vec<usize>>,
    ) -> Array2<f32>,
    max_distance: f32,
    cascade_depth: usize,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    let mut unmatched_detections = detection_indices.clone();
    let mut matches: Vec<Match> = vec![];

    for level in 0..cascade_depth {
        if unmatched_detections.is_empty() {
            // no detections left
            break;
        }

        let track_indices_l = track_indices
            .iter()
            .filter(|track_idx| tracks.get(**track_idx).unwrap().time_since_update == 1 + level)
            .cloned()
            .collect::<Vec<usize>>();
        if track_indices_l.is_empty() {
            // nothing to match at this level
            continue;
        }

        let (matches_l, _, unmatched_detections_l) = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            Some(track_indices_l),
            Some(detection_indices.to_owned()),
        );
        matches.extend_from_slice(&matches_l);
        unmatched_detections = unmatched_detections_l;
    }

    let unmatched_tracks = track_indices
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .difference(&matches.iter().map(|m| m.track_idx).collect::<HashSet<_>>())
        .collect::<Vec<&usize>>()
        .iter()
        .map(|v| *v.to_owned())
        .collect::<Vec<usize>>();

    (matches, unmatched_tracks, unmatched_detections)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    #[test]
    fn min_cost_matching() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[4.0, 5.0, 6.0, 7.0]));
        let t0 = Track::new(mean, covariance, 0, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[2.0, 3.0, 4.0, 5.0]));
        let t1 = Track::new(mean, covariance, 1, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[2.5, 3.5, 4.5, 5.5]));
        let t2 = Track::new(mean, covariance, 2, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[3.5, 4.5, 5.5, 6.5]));
        let t3 = Track::new(mean, covariance, 3, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[5.5, 6.5, 7.5, 8.5]));
        let t4 = Track::new(mean, covariance, 4, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[0.5, 1.5, 2.5, 3.5]));
        let t5 = Track::new(mean, covariance, 5, 0, 30, None);

        let d0 = Detection::new(arr1::<f32>(&[3.0, 4.0, 5.0, 6.0]), 1.0, arr1::<f32>(&[]));
        let d1 = Detection::new(arr1::<f32>(&[1.0, 2.0, 3.0, 4.0]), 1.0, arr1::<f32>(&[]));
        let d2 = Detection::new(arr1::<f32>(&[-17.0, 1.5, 42.0, 7.0]), 1.0, arr1::<f32>(&[]));
        let d3 = Detection::new(arr1::<f32>(&[-16.0, 2.0, 45.0, 6.0]), 1.0, arr1::<f32>(&[]));
        let d4 = Detection::new(
            arr1::<f32>(&[-12.5, -2.0, 30.0, 36.0]),
            1.0,
            arr1::<f32>(&[]),
        );
        let d5 = Detection::new(
            arr1::<f32>(&[-1.0, -12.5, 45.0, 45.0]),
            1.0,
            arr1::<f32>(&[]),
        );

        let (matches, unmatched_tracks, unmatched_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::iou_cost,
                0.7,
                &vec![t0, t1, t2, t3, t4, t5],
                &vec![d0, d1, d2, d3, d4, d5],
                None,
                None,
            );

        assert_eq!(matches, vec![Match::new(0, 2), Match::new(3, 3)]);
        assert_eq!(unmatched_tracks, vec![1, 2, 4, 5]);
        assert_eq!(unmatched_detections, vec![0, 1, 4, 5]);
    }

    #[test]
    fn matching_cascade() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[4.0, 5.0, 6.0, 7.0]));
        let mut t0 = Track::new(mean, covariance, 0, 0, 30, None);
        t0.time_since_update = 1;
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[2.0, 3.0, 4.0, 5.0]));
        let t1 = Track::new(mean, covariance, 1, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[2.5, 3.5, 4.5, 5.5]));
        let t2 = Track::new(mean, covariance, 2, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[3.5, 4.5, 5.5, 6.5]));
        let t3 = Track::new(mean, covariance, 3, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[5.5, 6.5, 7.5, 8.5]));
        let t4 = Track::new(mean, covariance, 4, 0, 30, None);
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[0.5, 1.5, 2.5, 3.5]));
        let t5 = Track::new(mean, covariance, 5, 0, 30, None);

        let d0 = Detection::new(arr1::<f32>(&[3.0, 4.0, 5.0, 6.0]), 1.0, arr1::<f32>(&[]));
        let d1 = Detection::new(arr1::<f32>(&[1.0, 2.0, 3.0, 4.0]), 1.0, arr1::<f32>(&[]));
        let d2 = Detection::new(arr1::<f32>(&[-17.0, 1.5, 42.0, 7.0]), 1.0, arr1::<f32>(&[]));
        let d3 = Detection::new(arr1::<f32>(&[-16.0, 2.0, 45.0, 6.0]), 1.0, arr1::<f32>(&[]));
        let d4 = Detection::new(
            arr1::<f32>(&[-12.5, -2.0, 30.0, 36.0]),
            1.0,
            arr1::<f32>(&[]),
        );
        let d5 = Detection::new(
            arr1::<f32>(&[-1.0, -12.5, 45.0, 45.0]),
            1.0,
            arr1::<f32>(&[]),
        );

        let (matches, mut unmatched_tracks, mut unmatched_detections) =
            linear_assignment::matching_cascade(
                iou_matching::iou_cost,
                0.7,
                30,
                &vec![t0, t1, t2, t3, t4, t5],
                &vec![d0, d1, d2, d3, d4, d5],
                None,
                None,
            );

        unmatched_tracks.sort_unstable();
        unmatched_detections.sort_unstable();

        assert_eq!(matches, vec![Match::new(0, 2)]);
        assert_eq!(unmatched_tracks, vec![1, 2, 3, 4, 5]);
        assert_eq!(unmatched_detections, vec![0, 1, 3, 4, 5]);
    }
}
