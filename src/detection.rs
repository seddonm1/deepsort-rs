use ndarray::*;

/**
This class represents a bounding box detection in a single image.

Parameters
----------
tlwh : array_like
    Bounding box in format `(x, y, w, h)`.
confidence : float
    Detector confidence score.
feature : array_like
    A feature vector that describes the object contained in this image.

Attributes
----------
tlwh : ndarray
    Bounding box in format `(top left x, top left y, width, height)`.
confidence : ndarray
    Detector confidence score.
feature : ndarray | NoneType
    A feature vector that describes the object contained in this image.
*/
#[derive(Debug, Clone)]
pub struct Detection {
    tlwh: Array1<f32>,
    confidence: f32,
    feature: Array1<f32>,
}

impl Detection {
    pub fn new(tlwh: Array1<f32>, confidence: f32, feature: Array1<f32>) -> Detection {
        Detection {
            tlwh,
            confidence,
            feature,
        }
    }

    /**
    Return the top, left, width, heigh of the detection
    */
    pub fn tlwh(&self) -> &Array1<f32> {
        &self.tlwh
    }

    /**
    Return the feature array of the detection
    */
    pub fn feature(&self) -> &Array1<f32> {
        &self.feature
    }

    /**
    Convert bounding box to format `(min x, min y, max x, max y)`, i.e., `(top left, bottom right)`.
    */
    pub fn to_tlbr(&self) -> Array1<f32> {
        array![
            self.tlwh[0],
            self.tlwh[1],
            self.tlwh[0] + self.tlwh[2],
            self.tlwh[1] + self.tlwh[3]
        ]
    }

    /**
    Convert bounding box to format `(center x, center y, aspect ratio, height)`, where the aspect ratio is `width / height`.
    */
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
