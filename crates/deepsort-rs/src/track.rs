use crate::*;
use anyhow::Result;
use ndarray::*;

/// Enumeration type for the single target track state:
///
/// - Newly created tracks are classified as `Tentative` until enough evidence has been collected.
/// - Then, the track state is changed to `Confirmed`.
/// - Tracks that are no longer alive are classified as `Deleted` to mark them for removal from the set of active tracks.
#[derive(Clone, Debug)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

/// Enumeration type for the source of the match
///
/// * `NearestNeighbor` means matched via the feature vector.
/// * `IoU` means matched via intsection over union of KalmanFilter predicted location.
#[derive(Clone, Debug)]
pub enum MatchSource {
    NearestNeighbor { detection: Detection, distance: f32 },
    IoU { detection: Detection, distance: f32 },
}

/// A single target track with state space `(x, y, a, h)` and associated velocities, where `(x, y)` is the center of the bounding box, `a` is the aspect ratio and `h` is the height.
#[derive(Clone, Debug)]
pub struct Track {
    /// The current track state.
    state: TrackState,
    /// Mean vector of the initial state distribution.
    mean: Array1<f32>,
    /// Covariance matrix of the initial state distribution.
    covariance: Array2<f32>,
    /// A unique track identifier.
    track_id: usize,
    /// The latest matched detection source.
    match_source: Option<MatchSource>,
    /// The last detection matched to this track
    detection: Detection,
    /// Number of consecutive detections before the track is confirmed
    n_init: usize,
    /// The maximum number of consecutive misses before the track state is set to `Deleted`.
    max_age: usize,
    /// Total number of measurement updates.
    hits: usize,
    /// Total number of frames since first occurance.
    age: usize,
    /// Total number of frames since last measurement update.
    time_since_update: usize,
    /// A cache of features. On each measurement update, the associated feature vector is added to this list.
    features: Option<Array2<f32>>,
}

impl Track {
    /// Returns a new Track
    ///
    /// # Parameters
    ///
    /// * `mean`: Mean vector of the initial state distribution.
    /// * `covariance`: Covariance matrix of the initial state distribution.
    /// * `track_id`: A unique track identifier.
    /// * `confidence`: A confidence score of the latest update.
    /// * `class_id`: An optional class identifier.
    /// * `n_init`: Number of consecutive detections before the track is confirmed. The track state is set to `Deleted` if a miss occurs within the first `n_init` frames.
    /// * `max_age`: The maximum number of consecutive misses before the track state is set to `Deleted`.
    /// * `features`: Feature vector of the detection this track originates from. If not None, this feature is added to the `features` cache.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mean: Array1<f32>,
        covariance: Array2<f32>,
        track_id: usize,
        detection: Detection,
        n_init: usize,
        max_age: usize,
        features: Option<Array2<f32>>,
    ) -> Track {
        Track {
            state: TrackState::Tentative,
            mean,
            covariance,
            track_id,
            match_source: None,
            detection,
            n_init,
            max_age,
            hits: 1,
            age: 1,
            time_since_update: 0,
            features,
        }
    }

    /// Return the identifier of the track
    pub fn track_id(&self) -> usize {
        self.track_id
    }

    /// Return the detection associated with the latest update
    pub fn detection(&self) -> &Detection {
        &self.detection
    }

    /// Return the TrackState of the track
    pub fn state(&self) -> &TrackState {
        &self.state
    }

    /// Return the match source of the track
    pub fn match_source(&self) -> &Option<MatchSource> {
        &self.match_source
    }

    /// Return the time since update of the track
    pub fn time_since_update(&self) -> usize {
        self.time_since_update
    }

    /// Return the mutable time since update of the track
    pub fn time_since_update_mut(&mut self) -> &mut usize {
        &mut self.time_since_update
    }

    /// Return the mean of the track
    pub fn mean(&self) -> &Array1<f32> {
        &self.mean
    }

    /// Return the covariance of the track
    pub fn covariance(&self) -> &Array2<f32> {
        &self.covariance
    }

    /// Return the features of the track
    pub fn features(&self) -> Option<&Array2<f32>> {
        self.features.as_ref()
    }

    /// Return the mutable features of the track
    pub fn features_mut(&mut self) -> &mut Option<Array2<f32>> {
        &mut self.features
    }

    /// Returns the track position bounding box
    pub fn bbox(&self) -> BoundingBox {
        let width = self.mean[2] * self.mean[3];
        let height = self.mean[3];
        let x = self.mean[0] - (width / 2.0);
        let y = self.mean[1] - (height / 2.0);
        BoundingBox::new(x, y, width, height)
    }

    /// Propagate the state distribution to the current time step using a Kalman filter prediction step.
    ///
    /// # Parameters
    ///
    /// * `kf`: The Kalman filter.
    pub fn predict(&mut self, kf: &KalmanFilter) {
        (self.mean, self.covariance) = kf.predict(&self.mean, &self.covariance);
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Perform Kalman filter measurement update step and update the feature cache.
    ///
    /// # Parameters
    ///
    /// * `kf`: The Kalman filter.
    /// * `detection`: The associated detection.
    pub fn update(
        &mut self,
        kf: &KalmanFilter,
        detection: &Detection,
        match_source: Option<MatchSource>,
    ) -> Result<()> {
        (self.mean, self.covariance) =
            kf.update(&self.mean, &self.covariance, &detection.bbox().to_xyah())?;

        self.match_source = match_source;
        self.detection = detection.clone();

        if let Some(feature) = detection.feature() {
            match &mut self.features {
                Some(features) => features.push_row(feature.view())?,
                None => self.features = Some(stack!(Axis(0), feature.to_owned())),
            };
        }

        self.hits += 1;
        self.time_since_update = 0;

        if matches!(self.state, TrackState::Tentative) && self.hits >= self.n_init {
            self.state = TrackState::Confirmed;
        }

        Ok(())
    }

    /// Mark this track as missed (no association at the current time step).
    pub fn mark_missed(&mut self) {
        if matches!(self.state, TrackState::Tentative) || self.time_since_update > self.max_age {
            self.state = TrackState::Deleted;
        }
    }

    /// Returns true if this track is tentative (unconfirmed).
    #[allow(dead_code)]
    pub fn is_tentative(&self) -> bool {
        matches!(self.state, TrackState::Tentative)
    }

    /// Returns true if this track is confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self.state, TrackState::Confirmed)
    }

    /// Returns true if this track is dead and should be deleted.
    pub fn is_deleted(&self) -> bool {
        matches!(self.state, TrackState::Deleted)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;
    use ndarray::*;
    use std::iter::FromIterator;

    #[test]
    fn to_tlwh() {
        let track = Track::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![[]],
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
            0,
            None,
        );
        assert_eq!(track.bbox().to_tlwh(), arr1::<f32>(&[-5.0, 0.0, 12.0, 4.0]));
    }

    #[test]
    fn to_tlbr() {
        let track = Track::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![[]],
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
            0,
            None,
        );
        assert_eq!(track.bbox().to_tlbr(), arr1::<f32>(&[-5.0, 0.0, 7.0, 4.0]));
    }

    #[test]
    fn predict() {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 2.0, 3.0, 4.0));

        let mut track = Track::new(
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
            0,
            None,
        );

        track.predict(&kf);

        assert!(track.is_tentative());
        assert_eq!(track.age, 2);
        assert_eq!(track.time_since_update, 1);
        assert_eq!(
            track.mean,
            arr1::<f32>(&[2.5, 4.0, 0.75, 4.0, 0.0, 0.0, 0.0, 0.0]),
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
    fn update() -> Result<()> {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 2.0, 3.0, 4.0));

        let mut track = Track::new(
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
            0,
            None,
        );
        track.predict(&kf);

        let detection = Detection::new(
            None,
            BoundingBox::new(2.0, 3.0, 4.0, 5.0),
            1.0,
            None,
            None,
            Some(Vec::<f32>::from_iter((0..128).map(|v| v as f32))),
        );
        track.update(&kf, &detection, None)?;

        assert!(track.is_confirmed());
        assert_eq!(track.hits, 2);
        assert_eq!(track.time_since_update, 0);
        assert_eq!(
            track.features,
            Some(stack![Axis(0), Array::range(0.0, 128.0, 1.0)])
        );
        assert_eq!(
            track.mean,
            arr1::<f32>(&[
                3.801653,
                5.301653,
                0.7509804,
                4.867769,
                0.30991733,
                0.30991733,
                0.00000000049019616,
                0.20661156
            ])
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

        Ok(())
    }
}
