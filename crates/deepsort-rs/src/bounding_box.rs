use ndarray::*;

/// BoundingBox represents the bounding box of the detection.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Left of the bounding box (i.e. min-x)
    x: f32,
    /// Top of the bounding box (i.e. max-y)
    y: f32,
    /// Width of the bounding box
    width: f32,
    /// Height of the bounding box
    height: f32,
}

impl PartialEq for BoundingBox {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.width == other.width
            && self.height == other.height
    }
}

impl BoundingBox {
    /// Returns a new BoundingBox
    ///
    /// # Parameters
    ///
    /// * `x`: Bounding box top.
    /// * `y`: Bounding box left.
    /// * `width`: Bounding box width.
    /// * `height`: Bounding box height.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> BoundingBox {
        BoundingBox {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns the x of the bounding box
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Returns the y of the bounding box
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Returns the width of the bounding box
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Returns the height of the bounding box
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Returns the area of the bounding box
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Returns the bounding box in left, top, width, height format, i.e., `(min x, min y, width, height)`.
    pub fn to_tlwh(&self) -> Array1<f32> {
        arr1::<f32>(&[self.x, self.y, self.width, self.height])
    }

    /// Returns the bounding box in top-left, bottom-right format, i.e., `(min x, min y, max x, max y)`.
    pub fn to_tlbr(&self) -> Array1<f32> {
        arr1::<f32>(&[self.x, self.y, self.x + self.width, self.y + self.height])
    }

    /// Returns the bounding box in center x, center y, aspect ratio, height format, where the aspect ratio is `width / height`.
    pub fn to_xyah(&self) -> Array1<f32> {
        arr1::<f32>(&[
            self.x + (self.width / 2.0),
            self.y + (self.height / 2.0),
            self.width / self.height,
            self.height,
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    #[test]
    fn to_tlwh() {
        let bbox = BoundingBox::new(1.0, 2.0, 13.0, 4.0);
        assert_eq!(bbox.to_tlwh(), arr1::<f32>(&[1.0, 2.0, 13.0, 4.0]));
    }

    #[test]
    fn to_tlbr() {
        let bbox = BoundingBox::new(1.0, 2.0, 13.0, 4.0);
        assert_eq!(bbox.to_tlbr(), arr1::<f32>(&[1.0, 2.0, 14.0, 6.0]));
    }

    #[test]
    fn to_xyah() {
        let bbox = BoundingBox::new(1.0, 2.0, 13.0, 4.0);
        assert_eq!(bbox.to_xyah(), arr1::<f32>(&[7.5, 4.0, 3.25, 4.0]));
    }
}
