use crate::*;
use anyhow::Result;
use kuhn_munkres::kuhn_munkres_min;
use ndarray::*;
use std::collections::HashSet;
use std::rc::Rc;

pub type DistanceMetricFn =
    Rc<dyn Fn(&[Track], &[Detection], &[usize], &[usize]) -> Result<Array2<f32>>>;

#[derive(Debug, Clone)]
pub struct Match {
    track_idx: usize,
    detection_idx: usize,
    distance: f32,
}

impl Match {
    /// Return a new Match
    ///
    /// # Parameters
    ///
    /// - `track_idx`: The match track index.
    /// - `detection_idx`: The match detection index.
    /// - `distance`: Match strength.
    pub fn new(track_idx: usize, detection_idx: usize, distance: f32) -> Match {
        Match {
            track_idx,
            detection_idx,
            distance,
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

    /// Return the distance of the match
    pub fn distance(&self) -> f32 {
        self.distance
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
/// - `distance_metric` : The distance metric is given a list of tracks and detections as well as a list of N track indices and M detection indices.
///   The metric should return the NxM dimensional cost matrix, where element (i, j) is the association cost between the i-th track in the given track indices and the j-th detection in the given detection_indices.
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
    distance_metric: DistanceMetricFn,
    max_distance: f32,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> Result<(Vec<Match>, Vec<usize>, Vec<usize>)> {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    if detection_indices.is_empty() || track_indices.is_empty() {
        Ok((vec![], track_indices, detection_indices))
    } else {
        let cost_matrix: Array2<f32> =
            (distance_metric)(tracks, detections, &track_indices, &detection_indices)?
                .mapv(|v| v.min(max_distance + 1e-5));

        // `kuhn_munkres_min` requires nrows() <= ncols() whereas scipy.optimize.linear_sum_assignment is able to process where nrows > ncols:
        // 'If it has more rows than columns, then not every row needs to be assigned to a column'
        //
        // to satisfy kuhn_munkres_min sequentially filter out the row with the lowest minimum value per row until nrows() <= ncols()
        // are no-op rows anyway
        let mut filtered_cost_matrix = Array2::<f32>::zeros((0, cost_matrix.ncols()));
        let mut filtered_indices: Vec<usize> = Vec::new();
        let (filtered_cost_matrix, filtered_indices) = if cost_matrix.nrows() > cost_matrix.ncols()
        {
            // collect the cost_matrix into a sequence of index, row, minimum value of the row
            let mut indexed_cost_matrix = cost_matrix
                .rows()
                .into_iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    (
                        row_idx,
                        row,
                        row.fold(f32::MAX, |accumulator, &value| accumulator.min(value)),
                    )
                })
                .collect::<Vec<(usize, ArrayView1<f32>, f32)>>();

            // sort by the minimum value for the row so that the worst match value is last can be `pop`ed from the vec
            indexed_cost_matrix.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

            // pop and disard the worst match rows until nrows == ncols keeping track of index
            while indexed_cost_matrix.len() > cost_matrix.ncols() {
                let (row_idx, _, _) = indexed_cost_matrix.pop().unwrap();
                filtered_indices.push(row_idx);
            }

            // re-sort by the index
            indexed_cost_matrix.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // push the remaining values into the matrix
            indexed_cost_matrix.iter().for_each(|(_, row, _)| {
                filtered_cost_matrix.push_row(*row).unwrap();
            });
            (&filtered_cost_matrix, filtered_indices)
        } else {
            (&cost_matrix, vec![])
        };

        // multiply by large constant to convert from f32 [0.0..1.0] to i64 which satisfies Matrix requirements (f32 does not implement `std::cmp::Ord`)
        let cost_vec = filtered_cost_matrix.mapv(|v| (v * 10_000_000_000.0) as i64);

        // invoke the kuhn munkres min (aka hungarian) assignment algorithm
        // this is equivalent to `scipy.optimize.linear_sum_assignment(maximise=False)` but where scipy returns two arrays
        // (row_ind and col_ind) `kuhn_munkres_min` returns just the col_ind array leaving row_ind (which is just a row index) to be
        // derived manually.
        let (_, col_indices) = kuhn_munkres_min(&cost_vec);
        let row_indices = (0..col_indices.len() + filtered_indices.len())
            .filter(|i| !filtered_indices.contains(i))
            .collect::<Vec<_>>();
        assert_eq!(row_indices.len(), col_indices.len());

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
                let distance = cost_matrix[[*row, *col]];

                if distance > max_distance {
                    unmatched_tracks.push(track_idx);
                    unmatched_detections.push(detection_idx);
                } else {
                    matches.push(Match::new(track_idx, detection_idx, distance));
                }
            });

        Ok((matches, unmatched_tracks, unmatched_detections))
    }
}

/// Run matching cascade.
///
/// # Parameters
///
/// - `distance_metric`: The distance metric is given a list of tracks and detections as well as a list of N track indices and M detection indices. The metric should return the NxM dimensional cost matrix, where element (i, j) is the association cost between the i-th track in the given track indices and the j-th detection in the given detection indices.
/// - `max_distance`: Gating threshold. Associations with cost larger than this value are disregarded.
/// - `cascade_depth`: The cascade depth, should be set to the maximum track age.
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
    distance_metric: DistanceMetricFn,
    max_distance: f32,
    cascade_depth: usize,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> Result<(Vec<Match>, Vec<usize>, Vec<usize>)> {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let unmatched_detections = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    // split into detections with and without feature vectors
    let (mut unmatched_detections, skipped_detections): (Vec<usize>, Vec<usize>) =
        unmatched_detections
            .into_iter()
            .partition(|detection_idx| detections.get(*detection_idx).unwrap().feature().is_some());

    let mut matches: Vec<Match> = vec![];

    for level in 0..cascade_depth {
        if unmatched_detections.is_empty() {
            // no detections left
            break;
        }

        let level_track_indices = track_indices
            .iter()
            .filter(|track_idx| tracks.get(**track_idx).unwrap().time_since_update() == 1 + level)
            .cloned()
            .collect::<Vec<_>>();

        if level_track_indices.is_empty() {
            // nothing to match at this level
            continue;
        }

        let (level_matches, _, level_unmatched_detections) = min_cost_matching(
            distance_metric.clone(),
            max_distance,
            tracks,
            detections,
            Some(level_track_indices),
            Some(unmatched_detections.to_owned()),
        )?;
        matches.extend_from_slice(&level_matches);
        unmatched_detections = level_unmatched_detections;
    }

    let match_indices = matches.iter().map(|m| m.track_idx).collect::<HashSet<_>>();
    let unmatched_tracks = track_indices
        .into_iter()
        .collect::<HashSet<_>>()
        .difference(&match_indices)
        .cloned()
        .collect::<Vec<_>>();

    Ok((
        matches,
        unmatched_tracks,
        [unmatched_detections, skipped_detections].concat(),
    ))
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
    kf: &KalmanFilter,
    mut cost_matrix: Array2<f32>,
    tracks: &[Track],
    detections: &[Detection],
    track_indices: &[usize],
    detection_indices: &[usize],
    gated_cost: Option<f32>,
    only_position: Option<bool>,
) -> Array2<f32> {
    let gated_cost = gated_cost.unwrap_or(f32::MAX);
    let gating_dim: usize = if only_position.unwrap_or(false) { 2 } else { 4 };
    let gating_threshold = kalman_filter::CHI2INV95.get(&gating_dim).unwrap();

    let measurements: Array2<f32> = stack(
        Axis(0),
        &detection_indices
            .iter()
            .map(|i| detections.get(*i).unwrap().bbox().to_xyah())
            .collect::<Vec<_>>()
            .iter()
            .map(|xyah| xyah.view())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    track_indices
        .iter()
        .enumerate()
        .for_each(|(row, track_idx)| {
            let track = tracks.get(*track_idx).unwrap();
            let gating_distance =
                kf.gating_distance(track.mean(), track.covariance(), &measurements);
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
    use std::iter::FromIterator;

    use crate::*;
    use ndarray::*;

    #[test]
    fn min_cost_matching() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 0.0, 5.0, 5.0));
        let t0 = Track::new(
            mean,
            covariance,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 1.0, 5.0, 5.0));
        let t1 = Track::new(
            mean,
            covariance,
            1,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(20.0, 20.0, 5.0, 5.0));
        let t2 = Track::new(
            mean,
            covariance,
            2,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );

        let d0 = Detection::new(
            None,
            BoundingBox::new(10.0, 10.0, 5.0, 5.0),
            1.0,
            None,
            None,
            None,
        );
        let d1 = Detection::new(
            None,
            BoundingBox::new(0.0, 0.0, 5.0, 5.0),
            1.0,
            None,
            None,
            None,
        );
        let d2 = Detection::new(
            None,
            BoundingBox::new(0.5, 0.5, 5.0, 5.0),
            1.0,
            None,
            None,
            None,
        );

        let (matches, unmatched_tracks, unmatched_detections) =
            linear_assignment::min_cost_matching(
                iou_matching::intersection_over_union_cost(),
                0.7,
                &[t0, t1, t2],
                &[d0, d1, d2],
                None,
                None,
            )
            .unwrap();

        assert_eq!(matches, vec![Match::new(0, 1, 1.0), Match::new(1, 2, 1.0)]);
        assert_eq!(unmatched_tracks, vec![2]);
        assert_eq!(unmatched_detections, vec![0]);
    }

    #[test]
    fn matching_cascade() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 0.0, 5.0, 5.0));
        let mut t0 = Track::new(
            mean,
            covariance,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 1.0, 5.0, 5.0));
        let t1 = Track::new(
            mean,
            covariance,
            1,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(20.0, 20.0, 5.0, 5.0));
        let t2 = Track::new(
            mean,
            covariance,
            2,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );

        let d0 = Detection::new(
            None,
            BoundingBox::new(10.0, 10.0, 5.0, 5.0),
            1.0,
            None,
            None,
            Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
        );
        let d1 = Detection::new(
            None,
            BoundingBox::new(0.0, 0.0, 5.0, 5.0),
            1.0,
            None,
            None,
            Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
        );
        let d2 = Detection::new(
            None,
            BoundingBox::new(0.5, 0.5, 5.0, 5.0),
            1.0,
            None,
            None,
            Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
        );

        *t0.time_since_update_mut() = 1;

        let (matches, mut unmatched_tracks, mut unmatched_detections) =
            linear_assignment::matching_cascade(
                iou_matching::intersection_over_union_cost(),
                0.7,
                30,
                &[t0, t1, t2],
                &[d0, d1, d2],
                None,
                None,
            )
            .unwrap();

        unmatched_tracks.sort_unstable();
        unmatched_detections.sort_unstable();

        assert_eq!(matches, vec![Match::new(0, 1, 1.0)]);
        assert_eq!(unmatched_tracks, vec![1, 2]);
        assert_eq!(unmatched_detections, vec![0, 2]);
    }

    #[test]
    fn gate_cost_matrix() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(4.0, 5.0, 6.0, 7.0));
        let t0 = Track::new(
            mean,
            covariance,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            0,
            30,
            None,
        );
        let d0 = Detection::new(
            None,
            BoundingBox::new(3.0, 4.0, 5.0, 6.0),
            1.0,
            None,
            None,
            None,
        );
        let d1 = Detection::new(
            None,
            BoundingBox::new(20.0, 20.0, 5.0, 6.0),
            1.0,
            None,
            None,
            None,
        );

        let cost_matrix = iou_matching::intersection_over_union_cost()(
            &vec![t0.clone()],
            &vec![d0.clone(), d1.clone()],
            &[0],
            &[0, 1],
        )
        .unwrap();

        let cost_matrix = linear_assignment::gate_cost_matrix(
            &kf,
            cost_matrix,
            &[t0],
            &[d0, d1],
            &[0],
            &[0],
            None,
            None,
        );
        assert_eq!(cost_matrix, arr2::<f32, _>(&[[0.6153846, 1.0]]));
    }
}
