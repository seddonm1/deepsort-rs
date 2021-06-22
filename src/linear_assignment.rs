use std::collections::HashSet;
use std::rc::Rc;

use crate::*;

use ndarray::*;
use pathfinding::kuhn_munkres::kuhn_munkres_min;
use pathfinding::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct Match {
    track_idx: usize,
    detection_idx: usize,
}

impl Match {
    /// Return a new Match
    ///
    /// # Parameters
    ///
    /// - `track_idx`: The match track index.
    /// - `detection_idx`: The match detection index.
    pub fn new(track_idx: usize, detection_idx: usize) -> Match {
        Match {
            track_idx,
            detection_idx,
        }
    }

    /// Return the track identifier of the match
    pub fn track_idx(&self) -> usize {
        self.track_idx
    }

    /// Return the detection identifier of the match
    pub fn detection_idx(&self) -> usize {
        self.detection_idx
    }
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.track_idx == other.track_idx && self.detection_idx == other.detection_idx
    }
}

/// Solve linear assignment problem.
///
/// # Parameters
///
/// - `distance_metric` : The distance metric is given a list of tracks and detections as well as a list of N track indices and M detection indices. The metric should return the NxM dimensional cost matrix, where element (i, j) is the association cost between the i-th track in the given track indices and the j-th detection in the given detection_indices.
/// - `max_distance`: Gating threshold. Associations with cost larger than this value are disregarded.
/// - `tracks`: A list of predicted tracks at the current time step.
/// - `detections`: A list of detections at the current time step.
/// - `track_indices`: List of track indices that maps rows in `cost_matrix` to tracks in `tracks` (see description above).
/// - `detection_indices`: List of detection indices that maps columns in `cost_matrix` to detections in `detections` (see description above).
///
/// # Returns
///
/// A tuple with the following three entries:
///
/// - A list of matched track and detection indices.
/// - A list of unmatched track indices.
/// - A list of unmatched detection indices.
#[allow(clippy::type_complexity)]
pub fn min_cost_matching(
    distance_metric: Rc<
        dyn Fn(&[Track], &[Detection], Option<Vec<usize>>, Option<Vec<usize>>) -> Array2<f32>,
    >,
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
        let cost_matrix: Array2<f32> = (distance_metric)(
            tracks,
            detections,
            Some(track_indices.clone()),
            Some(detection_indices.clone()),
        )
        .mapv(|v| v.min(max_distance + 1e-5));

        // scipy.optimize.linear_sum_assignment truncates rows to num columns:
        // 'If it has more rows than columns, then not every row needs to be assigned to a column'
        let cost_matrix =
            cost_matrix.slice(s![0..cost_matrix.nrows().min(cost_matrix.ncols()), ..]);

        // multiply by large constant to convert from f32 [0.0..1.0] to i64 which satisfies Matrix requirements (f32 does not implement `std::cmp::Ord`)
        let cost_vec = cost_matrix
            .mapv(|v| (v * 10_000_000_000.0) as i64)
            .iter()
            .cloned()
            .collect::<Vec<i64>>();

        // invoke the kuhn munkres min (aka hungarian) assignment algorithm
        // this is equivalent to `scipy.optimize.linear_sum_assignment(maximise=False)` but where scipy returns two arrays
        // (row_ind and col_ind) `kuhn_munkres_min` returns just the col_ind array leaving row_ind (which is just a row index) to be
        // derived manually.
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

/// Run matching cascade.
///
/// # Parameters
///
/// - `distance_metric`: The distance metric is given a list of tracks and detections as well as a list of N track indices and M detection indices. The metric should return the NxM dimensional cost matrix, where element (i, j) is the association cost between the i-th track in the given track indices and the j-th detection in the given detection indices.
/// - `max_distance`: Gating threshold. Associations with cost larger than this value are disregarded.
/// - `cascade_depth`: The cascade depth, should be se to the maximum track age.
/// - `tracks`: A list of predicted tracks at the current time step.
/// - `detections`: A list of detections at the current time step.
/// - `track_indices`: List of track indices that maps rows in `cost_matrix` to tracks in `tracks` (see description above). Defaults to all tracks.
/// - `detection_indices`: List of detection indices that maps columns in `cost_matrix` to detections in `detections` (see description above). Defaults to all detections.
///
/// # Returns
///
/// A tuple with the following three entries:
/// - A list of matched track and detection indices.
/// - A list of unmatched track indices.
/// - A list of unmatched detection indices.
#[allow(clippy::type_complexity)]
pub fn matching_cascade(
    distance_metric: Rc<
        dyn Fn(&[Track], &[Detection], Option<Vec<usize>>, Option<Vec<usize>>) -> Array2<f32>,
    >,
    max_distance: f32,
    cascade_depth: usize,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> (Vec<Match>, Vec<usize>, Vec<usize>) {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());

    let mut unmatched_detections =
        detection_indices.unwrap_or_else(|| (0..detections.len()).collect());
    let mut matches: Vec<Match> = vec![];

    for level in 0..cascade_depth {
        if unmatched_detections.is_empty() {
            // no detections left
            break;
        }

        let track_indices_l = track_indices
            .iter()
            .filter(|track_idx| *tracks.get(**track_idx).unwrap().time_since_update() == 1 + level)
            .cloned()
            .collect::<Vec<usize>>();
        if track_indices_l.is_empty() {
            // nothing to match at this level
            continue;
        }

        let (matches_l, _, unmatched_detections_l) = min_cost_matching(
            distance_metric.clone(),
            max_distance,
            tracks,
            detections,
            Some(track_indices_l),
            Some(unmatched_detections.to_owned()),
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

/// Invalidate infeasible entries in cost matrix based on the state distributions obtained by Kalman filtering.
///
/// # Parameters
///
/// - `kf`: The Kalman filter.
/// - `cost_matrix`: The NxM dimensional cost matrix, where N is the number of track indices and M is the number of detection indices, such that entry (i, j) is the association cost between `tracks[track_indices[i]]` and `detections[detection_indices[j]]`.
/// - `tracks`: A list of predicted tracks at the current time step.
/// - `detections`: A list of detections at the current time step.
/// - `track_indices`: List of track indices that maps rows in `cost_matrix` to tracks in `tracks` (see description above).
/// - `detection_indices`: List of detection indices that maps columns in `cost_matrix` to detections in `detections` (see description above).
/// - `gated_cost`: Entries in the cost matrix corresponding to infeasible associations are set this value. Defaults to a very large value.
/// - `only_position`: If true, only the x, y position of the state distribution is considered during gating. Defaults to false.
///
/// # Returns
///
/// The modified cost matrix.
#[allow(clippy::too_many_arguments)]
pub fn gate_cost_matrix(
    kf: KalmanFilter,
    mut cost_matrix: Array2<f32>,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Vec<usize>,
    detection_indices: Vec<usize>,
    gated_cost: Option<f32>,
    only_position: Option<bool>,
) -> Array2<f32> {
    let gated_cost = gated_cost.unwrap_or(f32::MAX);
    let gating_dim: usize = if only_position.unwrap_or(false) { 2 } else { 4 };
    let gating_threshold = kalman_filter::CHI2INV95.get(&gating_dim).unwrap();

    let mut measurements = Array2::zeros((0, 4));
    detection_indices.iter().for_each(|i| {
        measurements
            .push_row(detections.get(*i).unwrap().to_xyah().view())
            .unwrap()
    });

    track_indices
        .iter()
        .enumerate()
        .for_each(|(row, track_idx)| {
            let track = tracks.get(*track_idx).unwrap();
            let gating_distance =
                kf.gating_distance(&track.mean(), &track.covariance(), &measurements);
            gating_distance
                .iter()
                .enumerate()
                .for_each(|(i, gating_distance)| {
                    if gating_distance > gating_threshold {
                        cost_matrix[[row, i]] = gated_cost;
                    }
                });
        });

    cost_matrix
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

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

        let d0 = Detection::new(BoundingBox::new(3.0, 4.0, 5.0, 6.0), 1.0,  None);
        let d1 = Detection::new(BoundingBox::new(1.0, 2.0, 3.0, 4.0), 1.0,  None);
        let d2 = Detection::new(BoundingBox::new(-17.0, 1.5, 42.0, 7.0), 1.0,  None);
        let d3 = Detection::new(BoundingBox::new(-16.0, 2.0, 45.0, 6.0), 1.0, None);
        let d4 = Detection::new(BoundingBox::new(-12.5, -2.0, 30.0, 36.0), 1.0, None);
        let d5 = Detection::new(BoundingBox::new(-1.0, -12.5, 45.0, 45.0), 1.0,  None);

        let (matches, unmatched_tracks, unmatched_detections) =
            linear_assignment::min_cost_matching(
                Rc::new(iou_matching::iou_cost),
                0.7,
                &[t0, t1, t2, t3, t4, t5],
                &[d0, d1, d2, d3, d4, d5],
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
        *t0.time_since_update_mut() = 1;
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

        let d0 = Detection::new(BoundingBox::new(3.0, 4.0, 5.0, 6.0), 1.0, None);
        let d1 = Detection::new(BoundingBox::new(1.0, 2.0, 3.0, 4.0), 1.0,  None);
        let d2 = Detection::new(BoundingBox::new(-17.0, 1.5, 42.0, 7.0), 1.0, None);
        let d3 = Detection::new(BoundingBox::new(-16.0, 2.0, 45.0, 6.0), 1.0,  None);
        let d4 = Detection::new(BoundingBox::new(-12.5, -2.0, 30.0, 36.0), 1.0,  None);
        let d5 = Detection::new(BoundingBox::new(-1.0, -12.5, 45.0, 45.0), 1.0,  None);

        let (matches, mut unmatched_tracks, mut unmatched_detections) =
            linear_assignment::matching_cascade(
                Rc::new(iou_matching::iou_cost),
                0.7,
                30,
                &[t0, t1, t2, t3, t4, t5],
                &[d0, d1, d2, d3, d4, d5],
                None,
                None,
            );

        unmatched_tracks.sort_unstable();
        unmatched_detections.sort_unstable();

        assert_eq!(matches, vec![Match::new(0, 2)]);
        assert_eq!(unmatched_tracks, vec![1, 2, 3, 4, 5]);
        assert_eq!(unmatched_detections, vec![0, 1, 3, 4, 5]);
    }

    #[test]
    fn gate_cost_matrix() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[4.0, 5.0, 6.0, 7.0]));
        let t0 = Track::new(mean, covariance, 0, 0, 30, None);

        let d0 = Detection::new(BoundingBox::new(3.0, 4.0, 5.0, 6.0), 1.0,  None);

        let cost_matrix = linear_assignment::gate_cost_matrix(
            kf,
            arr2::<f32, _>(&[[0.2, 0.53]]),
            &[t0],
            &[d0],
            vec![0],
            vec![0],
            None,
            None,
        );
        assert_eq!(cost_matrix, arr2::<f32, _>(&[[f32::MAX, 0.53]]));
    }
}
