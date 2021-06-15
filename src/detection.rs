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
    Convert bounding box to format `(min x, min y, max x, max y)`, i.e., `(top left, bottom right)`.
    */
    fn to_tlbr(&self) -> Array1<f32> {
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
    fn to_xyah(&self) -> Array1<f32> {
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

        itertools::assert_equal(detection.to_tlbr(), array![1.0f32, 2.0f32, 4.0f32, 6.0f32]);
    }

    #[test]
    fn to_xyah() {
        let detection = Detection::new(array![1.0f32, 2.0f32, 3.0f32, 4.0f32], 1.0, array![]);

        itertools::assert_equal(detection.to_xyah(), array![2.5f32, 4.0f32, 0.75f32, 4.0f32]);
    }
}
