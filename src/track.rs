use ndarray::*;

use crate::Detection;
use crate::KalmanFilter;

/**
Enumeration type for the single target track state. Newly created tracks are
classified as `tentative` until enough evidence has been collected. Then,
the track state is changed to `confirmed`. Tracks that are no longer alive
are classified as `deleted` to mark them for removal from the set of active
tracks.
*/
#[derive(Debug)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

#[derive(Debug)]
pub struct Track {
    pub state: TrackState,
    pub mean: Array1<f32>,
    pub covariance: Array2<f32>,
    pub track_id: usize,
    n_init: usize,
    max_age: usize,
    hits: usize,
    age: usize,
    pub time_since_update: usize,
    pub features: Array2<f32>,
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
            features: feature.unwrap_or_else(|| Array2::zeros((0, 128))),
        }
    }

    pub fn to_tlwh(&self) -> Array1<f32> {
        let w = self.mean[2] * self.mean[3];
        let h = self.mean[3];
        let t = self.mean[0] - (w / 2.0);
        let l = self.mean[1] - (h / 2.0);
        arr1::<f32>(&[t, l, w, h])
    }

    pub fn to_tlbr(&self) -> Array1<f32> {
        let mut tlbr = self.to_tlwh();
        tlbr[2] += tlbr[0];
        tlbr[3] += tlbr[1];
        tlbr
    }

    /**
    Propagate the state distribution to the current time step using a
    Kalman filter prediction step.

    Parameters
    ----------
    kf : kalman_filter.KalmanFilter
        The Kalman filter.
    */
    pub fn predict(&mut self, kf: &KalmanFilter) {
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;
    }

    /**
    Perform Kalman filter measurement update step and update the feature
    cache.

    Parameters
    ----------
    kf : kalman_filter.KalmanFilter
        The Kalman filter.
    detection : Detection
        The associated detection.
    */
    pub fn update(&mut self, kf: &KalmanFilter, detection: &Detection) {
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, &detection.to_xyah());
        self.mean = mean;
        self.covariance = covariance;

        self.features.push_row(detection.feature.view()).unwrap();

        self.hits += 1;
        self.time_since_update = 0;

        if matches!(self.state, TrackState::Tentative) && self.hits >= self.n_init {
            self.state = TrackState::Confirmed
        }
    }

    /// Mark this track as missed (no association at the current time step).
    pub fn mark_missed(&mut self) {
        if matches!(self.state, TrackState::Tentative) || self.time_since_update > self.max_age {
            self.state = TrackState::Deleted;
        }
    }

    /// Returns True if this track is tentative (unconfirmed).
    #[allow(dead_code)]
    fn is_tentative(&self) -> bool {
        matches!(self.state, TrackState::Tentative)
    }

    /// Returns True if this track is confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self.state, TrackState::Confirmed)
    }

    /// Returns True if this track is dead and should be deleted.
    #[allow(dead_code)]
    fn is_deleted(&self) -> bool {
        matches!(self.state, TrackState::Deleted)
    }
}

#[cfg(test)]
mod tests {
    use crate::Detection;
    use crate::KalmanFilter;
    use crate::Track;
    use ndarray::*;

    #[test]
    fn to_tlwh() {
        let track = Track::new(array![1.0, 2.0, 3.0, 4.0], array![[]], 0, 0, 0, None);
        assert_eq!(track.to_tlwh(), array![-5.0f32, 0.0f32, 12.0f32, 4.0f32]);
    }

    #[test]
    fn to_tlbr() {
        let track = Track::new(array![1.0, 2.0, 3.0, 4.0], array![[]], 0, 0, 0, None);
        assert_eq!(track.to_tlbr(), array![-5.0f32, 0.0f32, 7.0f32, 4.0f32]);
    }

    #[test]
    fn predict() {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&array![1.0, 2.0, 3.0, 4.0]);

        let mut track = Track::new(mean, covariance, 0, 0, 0, None);

        track.predict(&kf);

        assert!(track.is_tentative());
        assert_eq!(track.age, 2);
        assert_eq!(track.time_since_update, 1);
        assert_eq!(
            track.mean,
            array![1.0f32, 2.0f32, 3.0f32, 4.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
        );
        assert_eq!(
            track.covariance,
            array![
                [
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0002000001f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.000000000099999994f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32
                ],
                [
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.000000000099999994f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000019999999f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32
                ]
            ],
        );
    }

    #[test]
    fn update() {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&array![1.0, 2.0, 3.0, 4.0]);

        let mut track = Track::new(mean, covariance, 0, 0, 0, None);
        track.predict(&kf);

        let detection = Detection::new(
            array![2.0, 3.0, 4.0, 5.0],
            1.0,
            Array::range(0.0, 128.0, 1.0),
        );
        track.update(&kf, &detection);

        assert!(track.is_confirmed());
        assert_eq!(track.hits, 2);
        assert_eq!(track.time_since_update, 0);
        assert_eq!(
            track.features,
            stack![Axis(0), Array::range(0.0, 128.0, 1.0)]
        );
        assert_eq!(
            track.mean,
            array![
                3.6033058f32,
                5.0371904f32,
                2.9568627f32,
                4.867769f32,
                0.61983466f32,
                0.7231405f32,
                -0.000000021568628f32,
                0.20661156f32
            ]
        );
        assert_eq!(
            track.covariance,
            array![
                [
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.00019607853f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000009803921f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32
                ],
                [
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.00000000009803921f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000019999999f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32
                ]
            ]
        );
    }
}
