use crate::BoundingBox;
use ndarray::*;

/// Detection represents a bounding box detection in a single image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in top, left, width, height format.
    bbox: BoundingBox,
    /// Detection confidence score.
    confidence: f32,
    /// Optional class identifier.
    class_id: Option<usize>,
    /// Optional class name
    class_name: Option<String>,
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
    /// - `class_id`: An optional class identifier.
    /// - `feature`: A feature vector that describes the object contained in this image.
    pub fn new(
        bbox: BoundingBox,
        confidence: f32,
        class_id: Option<usize>,
        class_name: Option<String>,
        feature: Option<Vec<f32>>,
    ) -> Detection {
        Detection {
            bbox,
            confidence,
            class_id,
            class_name,
            feature: feature.map(Array1::from_vec),
        }
    }

    /// Returns a BoundingBox of the detection co-ordinates
    pub fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Returns the confidence of the detection
    pub fn confidence(&self) -> &f32 {
        &self.confidence
    }

    /// Returns the class identifier of the detection
    pub fn class_id(&self) -> &Option<usize> {
        &self.class_id
    }

    /// Returns the class name of the detection
    pub fn class_name(&self) -> &Option<String> {
        &self.class_name
    }

    /// Returns the feature array of the detection
    pub fn feature(&self) -> &Option<Array1<f32>> {
        &self.feature
    }
}
