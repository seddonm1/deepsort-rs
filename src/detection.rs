use ndarray::*;

/// BoundingBox represents the bounding box of the detection.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Top of the bounding box (i.e. min-y)
    top: f32,
    /// Left of the bounding box (i.e. min-x)
    left: f32,
    /// Width of the bounding box
    width: f32,
    /// Height of the bounding box
    height: f32,
}

impl PartialEq for BoundingBox {
    fn eq(&self, other: &Self) -> bool {
        self.top == other.top
            && self.left == other.left
            && self.width == other.width
            && self.height == other.height
    }
}

impl BoundingBox {
    /// Returns a new BoundingBox
    ///
    /// # Parameters
    ///
    /// - `top`: Bounding box top.
    /// - `left`: Bounding box left.
    /// - `width`: Bounding box width.
    /// - `height`: Bounding box height.
    pub fn new(top: f32, left: f32, width: f32, height: f32) -> BoundingBox {
        assert!(width > 0.0, "width must be greater than 0");
        assert!(height > 0.0, "height must be greater than 0");
        BoundingBox {
            top,
            left,
            width,
            height,
        }
    }

    /// Returns the top of the bounding box
    pub fn top(&self) -> &f32 {
        &self.top
    }

    /// Returns the left of the bounding box
    pub fn left(&self) -> &f32 {
        &self.left
    }

    /// Returns the width of the bounding box
    pub fn width(&self) -> &f32 {
        &self.width
    }

    /// Returns the height of the bounding box
    pub fn height(&self) -> &f32 {
        &self.height
    }
}

/// Detection represents a bounding box detection in a single image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in top, left, width, height format.
    tlwh: Array1<f32>,
    /// Detection confidence score.
    confidence: f32,
    /// A feature vector that describes the object contained in this image.
    feature: Option<Array1<f32>>,
}

impl Detection {
    /// Returns a new Detection
    ///
    /// # Parameters
    ///
    /// - `bbox`: A bounding box object.
    /// - `confidence`: Detection confidence score.
    /// - `feature`: A feature vector that describes the object contained in this image.
    pub fn new(bbox: BoundingBox, confidence: f32, feature: Option<Vec<f32>>) -> Detection {
        Detection {
            tlwh: arr1::<f32>(&[bbox.top, bbox.left, bbox.width, bbox.height]),
            confidence,
            feature: feature.map(Array1::from_vec),
        }
    }

    /// Returns the detection bounding box in top, left, width, height format
    pub fn tlwh(&self) -> &Array1<f32> {
        &self.tlwh
    }

    /// Returns the feature array of the detection
    pub fn feature(&self) -> &Option<Array1<f32>> {
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

    /// Returns a BoundingBox of the detection co-ordinates
    pub fn to_bbox(&self) -> BoundingBox {
        BoundingBox::new(self.tlwh[0], self.tlwh[1], self.tlwh[2], self.tlwh[3])
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    #[test]
    fn to_tlbr() {
        let detection = Detection::new(BoundingBox::new(1.0, 2.0, 13.0, 4.0), 1.0,  None);

        assert_eq!(detection.to_tlbr(), arr1::<f32>(&[1.0, 2.0, 14.0, 6.0]));
    }

    #[test]
    fn to_xyah() {
        let detection = Detection::new(BoundingBox::new(1.0, 2.0, 13.0, 4.0), 1.0,  None);

        assert_eq!(detection.to_xyah(), arr1::<f32>(&[7.5, 4.0, 3.25, 4.0]));
    }

    #[test]
    fn to_bbox() {
        let bbox = BoundingBox::new(1.0, 2.0, 13.0, 4.0);
        let detection = Detection::new(bbox.clone(), 1.0, None);

        assert_eq!(detection.to_bbox(), bbox);
    }
}
