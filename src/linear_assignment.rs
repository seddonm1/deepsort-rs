use crate::*;

use ndarray::*;

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

// /**
// Solve linear assignment problem.

// Parameters
// ----------
// distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
//     The distance metric is given a list of tracks and detections as well as
//     a list of N track indices and M detection indices. The metric should
//     return the NxM dimensional cost matrix, where element (i, j) is the
//     association cost between the i-th track in the given track indices and
//     the j-th detection in the given detection_indices.
// max_distance : float
//     Gating threshold. Associations with cost larger than this value are
//     disregarded.
// tracks : List[track.Track]
//     A list of predicted tracks at the current time step.
// detections : List[detection.Detection]
//     A list of detections at the current time step.
// track_indices : List[int]
//     List of track indices that maps rows in `cost_matrix` to tracks in
//     `tracks` (see description above).
// detection_indices : List[int]
//     List of detection indices that maps columns in `cost_matrix` to
//     detections in `detections` (see description above).

// Returns
// -------
// (List[(int, int)], List[int], List[int])
//     Returns a tuple with the following three entries:
//     * A list of matched track and detection indices.
//     * A list of unmatched track indices.
//     * A list of unmatched detection indices.
// */
// pub fn min_cost_matching(distance_metric: fn(Vec<Track>, Vec<Detection>, Option<Vec<usize>>, Option<Vec<usize>>), max_distance: f32, tracks: Vec<Track>, detections: Vec<Detection>, track_indices: Option<Vec<usize>>, detection_indices: Option<Vec<usize>>) -> (Vec<Match>, Vec<usize>, Vec<usize>) {

//     let track_indices = track_indices.unwrap_or((0..tracks.len()).collect());
//     let detection_indices = detection_indices.unwrap_or((0..detections.len()).collect());

//     if detection_indices.len() == 0 || track_indices.len() == 0 {
//         // Nothing to match.
//         (vec![], track_indices, detection_indices)
//     } else {

//     }
// }

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn a() {
//     }

// }
