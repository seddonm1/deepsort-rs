use ndarray::*;

/// Detection represents a bounding box detection in a single image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in top, left, width, height format.
    tlwh: Array1<f32>,
    /// Detection confidence score.
    confidence: f32,
    /// A feature vector that describes the object contained in this image.
    feature: Array1<f32>,
}

impl Detection {
    /// Returns a new Detection
    ///
    /// # Parameters
    ///
    /// - `tlwh`: Bounding box in top, left, width, height format.
    /// - `confidence`: Detection confidence score.
    /// - `feature`: A feature vector that describes the object contained in this image.
    pub fn new(tlwh: Array1<f32>, confidence: f32, feature: Array1<f32>) -> Detection {
        assert!(tlwh.len() == 4);
        Detection {
            tlwh,
            confidence,
            feature,
        }
    }

    /// Returns the detection bounding box in top, left, width, height format
    pub fn tlwh(&self) -> &Array1<f32> {
        &self.tlwh
    }

    /// Returns the feature array of the detection
    pub fn feature(&self) -> &Array1<f32> {
        &self.feature
    }

    /// Returns the detection bounding box in top, left, bottom, right format, i.e., `(min x, min y, max x, max y)`.
    pub fn to_tlbr(&self) -> Array1<f32> {
        array![
            self.tlwh[0],
            self.tlwh[1],
            self.tlwh[0] + self.tlwh[2],
            self.tlwh[1] + self.tlwh[3]
        ]
    }

    /// Returns the detection bounding box in center x, center y, aspect ratio, height format, where the aspect ratio is `width / height`.
    pub fn to_xyah(&self) -> Array1<f32> {
        array![
            self.tlwh[0] + (self.tlwh[2] / 2.0),
            self.tlwh[1] + (self.tlwh[3] / 2.0),
            self.tlwh[2] / self.tlwh[3],
            self.tlwh[3]
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::Detection;
    use ndarray::*;

    #[test]
    fn to_tlbr() {
        let detection = Detection::new(array![1.0f32, 2.0f32, 3.0f32, 4.0f32], 1.0, array![]);

        assert_eq!(detection.to_tlbr(), arr1::<f32>(&[1.0, 2.0, 4.0, 6.0]));
    }

    #[test]
    fn to_xyah() {
        let detection = Detection::new(array![1.0f32, 2.0f32, 3.0f32, 4.0f32], 1.0, array![]);

        assert_eq!(detection.to_xyah(), arr1::<f32>(&[2.5, 4.0, 0.75, 4.0]));
    }
}
