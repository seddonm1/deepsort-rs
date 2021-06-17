use ndarray::*;

/**
Enumeration type for the single target track state. Newly created tracks are
classified as `tentative` until enough evidence has been collected. Then,
the track state is changed to `confirmed`. Tracks that are no longer alive
are classified as `deleted` to mark them for removal from the set of active
tracks.
*/
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

pub struct Track {
    state: TrackState,
    mean: Array1<f32>,
    covariance: Array2<f32>,
    track_id: usize,
    n_init: usize,
    max_age: usize,
    hits: usize,
    age: usize,
    time_since_update: usize,
    features: Array2<f32>,
}

/**
A single target track with state space `(x, y, a, h)` and associated
velocities, where `(x, y)` is the center of the bounding box, `a` is the
aspect ratio and `h` is the height.

Parameters
----------
mean : ndarray
    Mean vector of the initial state distribution.
covariance : ndarray
    Covariance matrix of the initial state distribution.
track_id : int
    A unique track identifier.
n_init : int
    Number of consecutive detections before the track is confirmed. The
    track state is set to `Deleted` if a miss occurs within the first
    `n_init` frames.
max_age : int
    The maximum number of consecutive misses before the track state is
    set to `Deleted`.
feature : Optional[ndarray]
    Feature vector of the detection this track originates from. If not None,
    this feature is added to the `features` cache.

Attributes
----------
mean : ndarray
    Mean vector of the initial state distribution.
covariance : ndarray
    Covariance matrix of the initial state distribution.
track_id : int
    A unique track identifier.
hits : int
    Total number of measurement updates.
age : int
    Total number of frames since first occurance.
time_since_update : int
    Total number of frames since last measurement update.
state : TrackState
    The current track state.
features : List[ndarray]
    A cache of features. On each measurement update, the associated feature
    vector is added to this list.
*/
impl Track {
    pub fn new(
        mean: Array1<f32>,
        covariance: Array2<f32>,
        track_id: usize,
        n_init: usize,
        max_age: usize,
        feature: Option<Array2<f32>>,
    ) -> Track {
        Track {
            state: TrackState::Tentative,
            mean,
            covariance,
            track_id,
            n_init,
            max_age,
            hits: 1,
            age: 1,
            time_since_update: 0,
            features: feature.unwrap_or(Array2::zeros((0, 128))),
        }
    }

    fn to_tlwh(&self) -> Array1<f32> {
        let w = self.mean[2] * self.mean[3];
        let h = self.mean[3];
        let t = self.mean[0] - (w / 2.0);
        let l = self.mean[1] - (h / 2.0);
        array![t, l, w, h]
    }

    fn to_tlbr(&self) -> Array1<f32> {
        let mut tlwh = self.to_tlwh();
        tlwh[2] = tlwh[2] + tlwh[0];
        tlwh[3] = tlwh[3] + tlwh[1];
        tlwh
    }

    /// Mark this track as missed (no association at the current time step).
    fn mark_missed(&mut self) {
        match self.state {
            TrackState::Tentative => {
                self.state = TrackState::Deleted;
            }
            _ => {
                if self.time_since_update > self.max_age {
                    self.state = TrackState::Deleted;
                }
            }
        }
    }

    /// Returns True if this track is tentative (unconfirmed).
    fn is_tentative(&self) -> bool {
        matches!(self.state, TrackState::Tentative)
    }

    /// Returns True if this track is confirmed.
    fn is_confirmed(&self) -> bool {
        matches!(self.state, TrackState::Confirmed)
    }

    /// Returns True if this track is dead and should be deleted.
    fn is_deleted(&self) -> bool {
        matches!(self.state, TrackState::Deleted)
    }
}

#[cfg(test)]
mod tests {
    use crate::Track;
    use ndarray::*;

    #[test]
    fn to_tlwh() {
        let t = Track::new(array![1.0, 2.0, 3.0, 4.0], array![[]], 0, 0, 0, None);
        assert_eq!(t.to_tlwh(), array![-5.0f32, 0.0f32, 12.0f32, 4.0f32]);
    }

    #[test]
    fn to_tlbr() {
        let t = Track::new(array![1.0, 2.0, 3.0, 4.0], array![[]], 0, 0, 0, None);
        assert_eq!(t.to_tlbr(), array![-5.0f32, 0.0f32, 7.0f32, 4.0f32]);
    }
}
