use anyhow::Result;
use std::num::NonZeroU32;
pub enum Interpolation {
  Nearest,
  Bilinear,
  Hamming,
  CatmullRom,
  Mitchell,
  Lanczos3,
}

impl From<Interpolation> for fast_image_resize::ResizeAlg {
  fn from(value: Interpolation) -> Self {
    match value {
      Interpolation::Nearest => fast_image_resize::ResizeAlg::Nearest,
      Interpolation::Bilinear => {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
      }
      Interpolation::Hamming => {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
      }
      Interpolation::CatmullRom => {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
      }
      Interpolation::Mitchell => {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
      }
      Interpolation::Lanczos3 => {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
      }
    }
  }
}

pub fn resize_image(
  image: &mut [u8],
  source_width: NonZeroU32,
  source_height: NonZeroU32,
  target_width: NonZeroU32,
  target_height: NonZeroU32,
  maintain_aspect_ratio: bool,
  interpolation: Interpolation,
) -> Result<Vec<u8>> {
  let source_image = fast_image_resize::Image::from_slice_u8(
    source_width,
    source_height,
    image,
    fast_image_resize::PixelType::U8x3,
  )?;

  // Create container for data of destination image. Initialised as RGB (0,0,0)
  let mut target_image =
    fast_image_resize::Image::new(target_width, target_height, source_image.pixel_type());

  // Get mutable view of destination image data
  let mut dst_view = if let Some(crop_box) = calculate_crop_box(
    source_width,
    source_height,
    target_width,
    target_height,
    maintain_aspect_ratio,
  ) {
    target_image.view_mut().crop(crop_box)?
  } else {
    target_image.view_mut()
  };

  // Create Resizer instance and resize source image
  // into buffer of destination image
  let mut resizer = fast_image_resize::Resizer::new(interpolation.into());

  // do the resize
  resizer.resize(&source_image.view(), &mut dst_view)?;

  Ok(target_image.into_vec())
}

/// Calculate padding
fn calculate_crop_box(
  source_width: NonZeroU32,
  source_height: NonZeroU32,
  target_width: NonZeroU32,
  target_height: NonZeroU32,
  maintain_aspect_ratio: bool,
) -> Option<fast_image_resize::CropBox> {
  let source_width = source_width.get() as f32;
  let source_height = source_height.get() as f32;
  let target_width = target_width.get();
  let target_height = target_height.get();

  let source_wh_ratio = source_width / source_height;
  let target_wh_ratio = target_width as f32 / target_height as f32;

  // apply simple scaling if padding is not relevant
  if !maintain_aspect_ratio || source_wh_ratio == target_wh_ratio {
    None
  } else {
    // otherwise calculate scaling on the longer axis
    let scale = (target_width as f32 / source_width).min(target_height as f32 / source_height);

    let scaled_width = (source_width * scale) as u32;
    let scaled_height = (source_height * scale) as u32;

    Some(fast_image_resize::CropBox {
      left: ((target_width - scaled_width) as f32 / 2.0) as u32,
      top: ((target_height - scaled_height) as f32 / 2.0) as u32,
      width: NonZeroU32::new(scaled_width).unwrap(),
      height: NonZeroU32::new(scaled_height).unwrap(),
    })
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use image::{ImageBuffer, Rgb};

  #[derive(Debug)]
  struct Case {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    maintain_aspect_ratio: bool,
    expected_x_pad: Option<u32>,
    expected_y_pad: Option<u32>,
  }

  #[test]
  fn test() {
    let test_cases = vec![
      Case {
        source_width: 640,
        source_height: 640,
        target_width: 1080,
        target_height: 1920,
        maintain_aspect_ratio: true,
        expected_x_pad: None,
        expected_y_pad: Some(420),
      },
      Case {
        source_width: 640,
        source_height: 640,
        target_width: 1920,
        target_height: 1080,
        maintain_aspect_ratio: true,
        expected_x_pad: Some(420),
        expected_y_pad: None,
      },
      Case {
        source_width: 720,
        source_height: 1080,
        target_width: 640,
        target_height: 640,
        maintain_aspect_ratio: true,
        expected_x_pad: Some(107),
        expected_y_pad: None,
      },
      Case {
        source_width: 1080,
        source_height: 720,
        target_width: 640,
        target_height: 640,
        maintain_aspect_ratio: true,
        expected_x_pad: None,
        expected_y_pad: Some(107),
      },
      Case {
        source_width: 1280,
        source_height: 720,
        target_width: 640,
        target_height: 384,
        maintain_aspect_ratio: true,
        expected_x_pad: None,
        expected_y_pad: Some(12),
      },
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
      let source_image = ImageBuffer::from_pixel(
        test_case.source_width,
        test_case.source_height,
        Rgb::from([255, 255, 255]),
      );

      let target_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
        test_case.target_width,
        test_case.target_height,
        resize_image(
          &mut source_image.to_vec(),
          NonZeroU32::new(test_case.source_width).unwrap(),
          NonZeroU32::new(test_case.source_height).unwrap(),
          NonZeroU32::new(test_case.target_width).unwrap(),
          NonZeroU32::new(test_case.target_height).unwrap(),
          test_case.maintain_aspect_ratio,
          Interpolation::Nearest,
        )
        .unwrap(),
      )
      .unwrap();

      // uncomment to write out images to verify scaling
      // target_image
      //     .save(format!(
      //         "{}_{}_{}_{}.gif",
      //         test_case.source_width,
      //         test_case.source_height,
      //         test_case.target_width,
      //         test_case.target_height,
      //     ))
      //     .unwrap();

      match test_case.expected_x_pad {
        Some(expected_x_pad) => {
          // test left
          assert_eq!(
            target_image.get_pixel(expected_x_pad - 1, 0),
            &Rgb::from([0, 0, 0]),
            "invalid left pad for case {i}"
          );
          assert_eq!(
            target_image.get_pixel(expected_x_pad, 0),
            &Rgb::from([255, 255, 255]),
            "invalid left pad for case {i}"
          );

          // test right
          assert_eq!(
            target_image.get_pixel(test_case.target_width - expected_x_pad, 0),
            &Rgb::from([0, 0, 0]),
            "invalid right pad for case {i}"
          );
          assert_eq!(
            target_image.get_pixel(test_case.target_width - expected_x_pad - 1, 0),
            &Rgb::from([255, 255, 255]),
            "invalid right pad for case {i}"
          );
        }
        None => {
          // test left
          assert_eq!(
            target_image.get_pixel(0, (test_case.target_height as f32 / 2.0) as u32),
            &Rgb::from([255, 255, 255]),
            "invalid left pad for case {i}"
          );
          // test right
          assert_eq!(
            target_image.get_pixel(
              test_case.target_width - 1,
              (test_case.target_height as f32 / 2.0) as u32
            ),
            &Rgb::from([255, 255, 255]),
            "invalid right pad for case {i}"
          );
        }
      }

      match test_case.expected_y_pad {
        Some(expected_y_pad) => {
          // test top
          assert_eq!(
            target_image.get_pixel(0, expected_y_pad - 1),
            &Rgb::from([0, 0, 0]),
            "invalid top pad for case {i}"
          );
          assert_eq!(
            target_image.get_pixel(0, expected_y_pad),
            &Rgb::from([255, 255, 255]),
            "invalid top pad for case {i}"
          );

          // test bottom
          assert_eq!(
            target_image.get_pixel(0, test_case.target_height - expected_y_pad),
            &Rgb::from([0, 0, 0]),
            "invalid bottom pad for case {i}"
          );
          assert_eq!(
            target_image.get_pixel(0, test_case.target_height - expected_y_pad - 1),
            &Rgb::from([255, 255, 255]),
            "invalid bottom pad for case {i}"
          );
        }
        None => {
          // test top
          assert_eq!(
            target_image.get_pixel((test_case.target_width as f32 / 2.0) as u32, 0),
            &Rgb::from([255, 255, 255]),
            "invalid top pad for case {i}"
          );
          // test bottom
          assert_eq!(
            target_image.get_pixel(
              (test_case.target_width as f32 / 2.0) as u32,
              test_case.target_height - 1
            ),
            &Rgb::from([255, 255, 255]),
            "invalid bottom pad for case {i}"
          );
        }
      }
    }
  }
}
