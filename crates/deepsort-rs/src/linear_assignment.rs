use crate::*;
use anyhow::{Ok, Result};
use kuhn_munkres::kuhn_munkres_min;
use ndarray::*;
use std::collections::HashMap;
use std::rc::Rc;

pub type DistanceMetricFn = Rc<dyn Fn(&[Track], &[Detection]) -> Result<Array2<f32>>>;

#[derive(Debug)]
pub struct Match {
    pub track: Track,
    pub detection: Detection,
    pub distance: f32,
}

impl Match {
    /// Return a new Match
    ///
    /// # Parameters
    ///
    /// * `track_idx`: The match track index.
    /// * `detection_idx`: The match detection index.
    /// * `distance`: Match strength.
    pub fn new(track: Track, detection: Detection, distance: f32) -> Match {
        Match {
            track,
            detection,
            distance,
        }
    }

    /// Return the distance of the match
    pub fn distance(&self) -> f32 {
        self.distance
    }
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.track == other.track && self.detection == other.detection
    }
}

/// Solve linear assignment problem.
///
/// # Parameters
///
/// * `distance_metric` : The distance metric is given a list of tracks and detections as well as a list of N track indices and M detection indices.
///   The metric should return the NxM dimensional cost matrix, where element (i, j) is the association cost between the i-th track in the given track indices and the j-th detection in the given detection_indices.
/// * `max_distance`: Gating threshold. Associations with cost larger than this value are disregarded.
/// * `tracks`: A list of predicted tracks at the current time step.
/// * `detections`: A list of detections at the current time step.
/// * `track_indices`: List of track indices that maps rows in `cost_matrix` to tracks in `tracks` (see description above).
/// * `detection_indices`: List of detection indices that maps columns in `cost_matrix` to detections in `detections` (see description above).
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
    tracks: Vec<Track>,
    detections: Vec<Detection>,
) -> Result<(Vec<Match>, Vec<Track>, Vec<Detection>)> {
    if tracks.is_empty() || detections.is_empty() {
        Ok((vec![], tracks, detections))
    } else {
        let cost_matrix: Array2<f32> =
            (distance_metric)(&tracks, &detections)?.mapv(|v| v.min(max_distance + 1e-5));

        let mut tracks = tracks.into_iter().enumerate().collect::<HashMap<_, _>>();
        let mut detections = detections
            .into_iter()
            .enumerate()
            .collect::<HashMap<_, _>>();

        // convert
        let (cost_matrix, transposed) = if cost_matrix.nrows() > cost_matrix.ncols() {
            (cost_matrix.t(), true)
        } else {
            (cost_matrix.view(), false)
        };

        // multiply by large constant to convert from f32 [0.0..1.0] to i64 which satisfies Matrix requirements (f32 does not implement `std::cmp::Ord`)
        let cost_vec = cost_matrix.mapv(|v| (v * 10_000_000_000.0) as i64);

        // invoke the kuhn munkres min (aka hungarian) assignment algorithm
        // this is equivalent to `scipy.optimize.linear_sum_assignment(maximise=False)` but where scipy returns two arrays
        // (row_ind and col_ind) `kuhn_munkres_min` returns just the col_ind array leaving row_ind (which is just a row index) to be
        // derived manually.
        let (_, col_indices) = kuhn_munkres_min(&cost_vec);
        let row_indices = (0..col_indices.len()).collect::<Vec<_>>();

        let mut matches: Vec<Match> = Vec::with_capacity(tracks.len().max(detections.len()));
        row_indices
            .into_iter()
            .zip(col_indices.into_iter())
            .for_each(|(row, col)| {
                let distance = cost_matrix[[row, col]];
                if distance < max_distance {
                    let track = tracks.get(if transposed { &col } else { &row }).unwrap();
                    let detection = detections
                        .get(if transposed { &row } else { &col })
                        .unwrap();
                    println!("MATCH {:?} {:?} {}", track, detection, distance);

                    let track = tracks.remove(if transposed { &col } else { &row }).unwrap();
                    let detection = detections
                        .remove(if transposed { &row } else { &col })
                        .unwrap();
                    let m = Match::new(track, detection, distance);
                    // println!("{:?}", m);
                    matches.push(m);
                } else {
                    let track = tracks.get(if transposed { &col } else { &row }).unwrap();
                    let detection = detections
                        .get(if transposed { &row } else { &col })
                        .unwrap();
                    println!("NOT MATCH {:?} {:?} {}", track, detection, distance);
                }
            });

        Ok((
            matches,
            tracks.into_values().collect::<Vec<_>>(),
            detections.into_values().collect::<Vec<_>>(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;
    use uuid::Uuid;

    #[test]
    fn min_cost_matching() -> Result<()> {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 0.0, 5.0, 5.0));
        let t0 = Track::new(
            track::TrackState::Tracked,
            true,
            mean,
            covariance,
            0,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 1.0, 5.0, 5.0));
        let t1 = Track::new(
            track::TrackState::Tracked,
            true,
            mean,
            covariance,
            1,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            None,
        );
        let (mean, covariance) = kf.initiate(&BoundingBox::new(20.0, 20.0, 5.0, 5.0));
        let t2 = Track::new(
            track::TrackState::Tracked,
            true,
            mean,
            covariance,
            2,
            0,
            Detection::new(
                None,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0),
                1.0,
                None,
                None,
                None,
            ),
            None,
        );

        let d0 = Detection::new(
            Some(Uuid::parse_str("47cd553d-d12f-4d2e-904b-0004d631fd6d").unwrap()),
            BoundingBox::new(10.0, 10.0, 5.0, 5.0),
            1.0,
            None,
            None,
            None,
        );
        let d1 = Detection::new(
            Some(Uuid::parse_str("b5b492b9-d14b-49d6-8792-033fb876e5eb").unwrap()),
            BoundingBox::new(0.0, 0.0, 5.0, 5.0),
            1.0,
            None,
            None,
            None,
        );
        let d2 = Detection::new(
            Some(Uuid::parse_str("3cb19956-7222-409a-9a36-aa4ce795862c").unwrap()),
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
                vec![t0, t1, t2],
                vec![d0, d1, d2],
            )?;

        matches
            .into_iter()
            .enumerate()
            .for_each(|(match_idx, r#match)| {
                let Match {
                    track, detection, ..
                } = r#match;

                match match_idx {
                    0 => {
                        assert_eq!(track.track_id(), 0);
                        assert_eq!(
                            detection.id(),
                            &Uuid::parse_str("b5b492b9-d14b-49d6-8792-033fb876e5eb").unwrap()
                        );
                    }
                    1 => {
                        assert_eq!(track.track_id(), 1);
                        assert_eq!(
                            detection.id(),
                            &Uuid::parse_str("3cb19956-7222-409a-9a36-aa4ce795862c").unwrap()
                        );
                    }
                    _ => unimplemented!(),
                }
            });

        assert_eq!(unmatched_tracks.get(0).unwrap().track_id(), 2);
        assert_eq!(
            unmatched_detections.get(0).unwrap().id(),
            &Uuid::parse_str("47cd553d-d12f-4d2e-904b-0004d631fd6d").unwrap()
        );

        Ok(())
    }
}

//     #[test]
//     fn matching_cascade() -> Result<()> {
//         let kf = KalmanFilter::new();

//         let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 0.0, 5.0, 5.0));
//         let mut t0 = Track::new(
//             mean,
//             covariance,
//             0,
//             Detection::new(
//                 None,
//                 BoundingBox::new(0.0, 0.0, 0.0, 0.0),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             ),
//             0,
//             30,
//             None,
//         );
//         let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 1.0, 5.0, 5.0));
//         let t1 = Track::new(
//             mean,
//             covariance,
//             1,
//             Detection::new(
//                 None,
//                 BoundingBox::new(0.0, 0.0, 0.0, 0.0),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             ),
//             0,
//             30,
//             None,
//         );
//         let (mean, covariance) = kf.initiate(&BoundingBox::new(20.0, 20.0, 5.0, 5.0));
//         let t2 = Track::new(
//             mean,
//             covariance,
//             2,
//             Detection::new(
//                 None,
//                 BoundingBox::new(0.0, 0.0, 0.0, 0.0),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             ),
//             0,
//             30,
//             None,
//         );

//         let d0 = Detection::new(
//             None,
//             BoundingBox::new(10.0, 10.0, 5.0, 5.0),
//             1.0,
//             None,
//             None,
//             Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
//         );
//         let d1 = Detection::new(
//             None,
//             BoundingBox::new(0.0, 0.0, 5.0, 5.0),
//             1.0,
//             None,
//             None,
//             Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
//         );
//         let d2 = Detection::new(
//             None,
//             BoundingBox::new(0.5, 0.5, 5.0, 5.0),
//             1.0,
//             None,
//             None,
//             Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
//         );

//         *t0.time_since_update_mut() = 1;

//         let (matches, mut unmatched_tracks, mut unmatched_detections) =
//             linear_assignment::matching_cascade(
//                 iou_matching::intersection_over_union_cost(),
//                 0.7,
//                 30,
//                 &[t0, t1, t2],
//                 &[d0, d1, d2],
//                 None,
//                 None,
//             )?;

//         unmatched_tracks.sort_unstable();
//         unmatched_detections.sort_unstable();

//         assert_eq!(matches, vec![Match::new(0, 1, 1.0)]);
//         assert_eq!(unmatched_tracks, vec![1, 2]);
//         assert_eq!(unmatched_detections, vec![0, 2]);

//         Ok(())
//     }

//     #[test]
//     fn gate_cost_matrix() -> Result<()> {
//         let kf = KalmanFilter::new();

//         let (mean, covariance) = kf.initiate(&BoundingBox::new(4.0, 5.0, 6.0, 7.0));
//         let t0 = Track::new(
//             mean,
//             covariance,
//             0,
//             Detection::new(
//                 None,
//                 BoundingBox::new(0.0, 0.0, 0.0, 0.0),
//                 1.0,
//                 None,
//                 None,
//                 None,
//             ),
//             0,
//             30,
//             None,
//         );
//         let d0 = Detection::new(
//             None,
//             BoundingBox::new(3.0, 4.0, 5.0, 6.0),
//             1.0,
//             None,
//             None,
//             None,
//         );
//         let d1 = Detection::new(
//             None,
//             BoundingBox::new(20.0, 20.0, 5.0, 6.0),
//             1.0,
//             None,
//             None,
//             None,
//         );

//         let cost_matrix = iou_matching::intersection_over_union_cost()(
//             &vec![t0.clone()],
//             &vec![d0.clone(), d1.clone()],
//             &[0],
//             &[0, 1],
//         )?;

//         let cost_matrix = linear_assignment::gate_cost_matrix(
//             &kf,
//             cost_matrix,
//             &[t0],
//             &[d0, d1],
//             &[0],
//             &[0],
//             None,
//             None,
//         )?;
//         assert_eq!(cost_matrix, arr2::<f32, _>(&[[0.6153846, 1.0]]));

//         Ok(())
//     }
// }
