use std::cmp::Ordering;

/// Translate / rescale an (x , y) coordinate from the source width/height to the target width/height.
/// If source_padded is set, the padding will be removed.
pub fn translate(
  mut x: f32,
  mut y: f32,
  source_width: u32,
  source_height: u32,
  target_width: u32,
  target_height: u32,
  source_padded: bool,
) -> (f32, f32) {
  let source_width = source_width as f32;
  let source_height = source_height as f32;
  let target_width = target_width as f32;
  let target_height = target_height as f32;

  let scale_x = target_width / source_width;
  let scale_y = target_height / source_height;

  let source_hw_ratio = source_height / source_width;
  let target_hw_ratio = target_height / target_width;

  let source_wh_ratio = source_width / source_height;
  let target_wh_ratio = target_width / target_height;

  // apply simple scaling if padding is not relevant
  if !source_padded || source_hw_ratio == target_hw_ratio {
    x *= scale_x;
    y *= scale_y;
  } else {
    // scale and subtract the (scaled) padding from the correct dimension
    match target_width.partial_cmp(&target_height) {
      Some(ord) => match ord {
        // maybe left and right padded
        Ordering::Less => {
          // scale both by longer dimension
          x *= scale_y;
          y *= scale_y;

          // subtract padding from the padded x dimension
          let padding =
            (source_width - source_width / target_hw_ratio * source_hw_ratio) / 2.0 * scale_y;
          x -= padding;
        }
        Ordering::Equal => match source_width.partial_cmp(&source_height) {
          Some(cmp) => match cmp {
            Ordering::Less => {
              // scale both by longer dimension
              x *= scale_x;
              y *= scale_x;

              // // subtract padding from the padded y dimension
              let padding =
                (source_height - source_height / target_wh_ratio * source_wh_ratio) / 2.0 * scale_x;
              y -= padding;
            }
            Ordering::Equal => unreachable!(),
            Ordering::Greater => {
              // scale both by longer dimension
              x *= scale_y;
              y *= scale_y;

              // subtract padding from the padded x dimension
              let padding =
                (source_width - source_width / target_hw_ratio * source_hw_ratio) / 2.0 * scale_y;
              x -= padding;
            }
          },
          None => unimplemented!(),
        },
        // maybe top and bottom padded
        Ordering::Greater => {
          // scale both by longer dimension
          x *= scale_x;
          y *= scale_x;

          // subtract padding from the padded y dimension
          let padding =
            (source_height - source_height / target_wh_ratio * source_wh_ratio) / 2.0 * scale_x;
          y -= padding;
        }
      },
      None => unimplemented!(),
    }
  }
  (x, y)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[derive(Debug)]
  struct Case {
    x: f32,
    y: f32,
    source_width: u32,
    source_height: u32,
    maintain_aspect_ratio: bool,
    target_width: u32,
    target_height: u32,
    expected_x: f32,
    expected_y: f32,
  }

  #[test]
  fn test_translate_no_padding() {
    let no_padding_cases = vec![
      // middle
      Case {
        x: 320.0,
        y: 320.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 540.0,
      },
      // top left
      Case {
        x: 0.0,
        y: 0.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 320.0,
        y: 0.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 640.0,
        y: 0.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 1920.0,
        expected_y: 0.0,
      },
      // middle right
      Case {
        x: 640.0,
        y: 320.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 1920.0,
        expected_y: 540.0,
      },
      // bottom right
      Case {
        x: 640.0,
        y: 640.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 2560,
        target_height: 1440,
        expected_x: 2560.0,
        expected_y: 1440.0,
      },
      // middle bottom
      Case {
        x: 320.0,
        y: 640.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 1080.0,
      },
      // bottom left
      Case {
        x: 0.0,
        y: 640.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 1080.0,
      },
      // middle left
      Case {
        x: 0.0,
        y: 320.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: false,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 540.0,
      },
    ];

    for test_case in no_padding_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(x, test_case.expected_x, "no_padding_cases: {test_case:?}");
      assert_eq!(y, test_case.expected_y, "no_padding_cases: {test_case:?}");
    }
  }

  #[test]
  fn test_translate_matching_aspect_ratios() {
    let matching_aspect_ratios_cases = vec![
      // middle
      Case {
        x: 960.0,
        y: 540.0,
        source_width: 1920,
        source_height: 1080,
        maintain_aspect_ratio: true,
        target_width: 2560,
        target_height: 1440,
        expected_x: 1280.0,
        expected_y: 720.0,
      },
      // top left
      Case {
        x: 0.0,
        y: 0.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 960.0,
        y: 0.0,
        source_width: 1920,
        source_height: 1080,
        maintain_aspect_ratio: true,
        target_width: 2560,
        target_height: 1440,
        expected_x: 1280.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 0.0,
        y: 0.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // middle right
      Case {
        x: 1280.0,
        y: 720.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 540.0,
      },
      // bottom right
      Case {
        x: 2560.0,
        y: 1440.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 1920.0,
        expected_y: 1080.0,
      },
      // bottom middle
      Case {
        x: 1280.0,
        y: 1440.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 1080.0,
      },
      // bottom left
      Case {
        x: 0.0,
        y: 500.0,
        source_width: 1000,
        source_height: 500,
        maintain_aspect_ratio: true,
        target_width: 540,
        target_height: 270,
        expected_x: 0.0,
        expected_y: 270.0,
      },
      // middle left
      Case {
        x: 0.0,
        y: 720.0,
        source_width: 2560,
        source_height: 1440,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 540.0,
      },
    ];

    for test_case in matching_aspect_ratios_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(
        x, test_case.expected_x,
        "matching_aspect_ratio_cases: {test_case:?}"
      );
      assert_eq!(
        y, test_case.expected_y,
        "matching_aspect_ratio_cases: {test_case:?}"
      );
    }
  }

  #[test]
  fn test_translate_wide_target_with_padding() {
    let wide_target_with_padding_cases = vec![
      // middle
      Case {
        x: 320.0,
        y: 320.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 2560,
        target_height: 1440,
        expected_x: 1280.0,
        expected_y: 720.0,
      },
      // top left
      Case {
        x: 0.0,
        y: 140.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 320.0,
        y: 140.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 640.0,
        y: 160.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 1000,
        target_height: 500,
        expected_x: 1000.0,
        expected_y: 0.0,
      },
      // middle right
      Case {
        x: 1024.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 1920.0,
        expected_y: 540.0,
      },
      // bottom right
      Case {
        x: 640.0,
        y: 576.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 500,
        target_height: 400,
        expected_x: 500.0,
        expected_y: 400.0,
      },
      // bottom middle
      Case {
        x: 320.0,
        y: 500.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 960.0,
        expected_y: 1080.0,
      },
      // bottom left
      Case {
        x: 0.0,
        y: 500.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 2560,
        target_height: 1440,
        expected_x: 0.0,
        expected_y: 1440.0,
      },
      // middle left
      Case {
        x: 0.0,
        y: 320.0,
        source_width: 640,
        source_height: 640,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1080,
        expected_x: 0.0,
        expected_y: 540.0,
      },
    ];

    for test_case in wide_target_with_padding_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(
        x, test_case.expected_x,
        "wide_target_with_padding_cases: {test_case:?}"
      );
      assert_eq!(
        y, test_case.expected_y,
        "wide_target_with_padding_cases: {test_case:?}"
      );
    }
  }

  #[test]
  fn test_translate_tall_target_with_padding() {
    let tall_target_with_padding_cases = vec![
      // middle
      Case {
        x: 512.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 540.0,
        expected_y: 960.0,
      },
      // top left
      Case {
        x: 296.0,
        y: 0.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 512.0,
        y: 0.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 540.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 820.0,
        y: 0.0,
        source_width: 1024,
        source_height: 770,
        maintain_aspect_ratio: true,
        target_width: 400,
        target_height: 500,
        expected_x: 400.0,
        expected_y: 0.0,
      },
      // middle right
      Case {
        x: 728.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 1080.0,
        expected_y: 960.0,
      },
      // bottom right
      Case {
        x: 728.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 540,
        target_height: 960,
        expected_x: 540.0,
        expected_y: 960.0,
      },
      // bottom middle
      Case {
        x: 512.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 540.0,
        expected_y: 1920.0,
      },
      // bottom left
      Case {
        x: 296.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 1920.0,
      },
      // middle left
      Case {
        x: 296.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1080,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 960.0,
      },
    ];
    for test_case in tall_target_with_padding_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(
        x, test_case.expected_x,
        "tall_target_with_padding_cases: {test_case:?}"
      );
      assert_eq!(
        y, test_case.expected_y,
        "tall_target_with_padding_cases: {test_case:?}"
      );
    }
  }

  #[test]
  fn test_translate_square_target_wide_source_with_padding() {
    let square_target_wide_source_with_padding_cases = vec![
      // middle
      Case {
        x: 512.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 960.0,
        expected_y: 960.0,
      },
      // top left
      Case {
        x: 128.0,
        y: 0.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 512.0,
        y: 0.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 2560,
        target_height: 2560,
        expected_x: 1280.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 450.0,
        y: 0.0,
        source_width: 500,
        source_height: 400,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 0.0,
      },
      // middle right
      Case {
        x: 896.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 960.0,
      },
      // bottom right
      Case {
        x: 896.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 1920.0,
      },
      // bottom middle
      Case {
        x: 512.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 960.0,
        expected_y: 1920.0,
      },
      // bottom left
      Case {
        x: 128.0,
        y: 768.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 1920.0,
      },
      // middle left
      Case {
        x: 128.0,
        y: 384.0,
        source_width: 1024,
        source_height: 768,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 960.0,
      },
    ];

    for test_case in square_target_wide_source_with_padding_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(
        x, test_case.expected_x,
        "square_target_wide_source_with_padding_cases: {test_case:?}"
      );
      assert_eq!(
        y, test_case.expected_y,
        "square_target_wide_source_with_padding_cases: {test_case:?}"
      );
    }
  }

  #[test]
  fn test_translate_square_target_tall_source_with_padding() {
    let square_target_tall_source_with_padding_cases = vec![
      // top left
      Case {
        x: 0.0,
        y: 128.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 0.0,
      },
      // top middle
      Case {
        x: 384.0,
        y: 128.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 960.0,
        expected_y: 0.0,
      },
      // top right
      Case {
        x: 768.0,
        y: 128.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 0.0,
      },
      // right middle
      Case {
        x: 768.0,
        y: 512.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 960.0,
      },
      // bottom right
      Case {
        x: 768.0,
        y: 896.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 1920.0,
      },
      // bottom middle
      Case {
        x: 384.0,
        y: 896.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 960.0,
        expected_y: 1920.0,
      },
      //  bottom left
      Case {
        x: 0.0,
        y: 600.0,
        source_width: 300,
        source_height: 900,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 0.0,
        expected_y: 1920.0,
      },
      // left middle
      Case {
        x: 768.0,
        y: 512.0,
        source_width: 768,
        source_height: 1024,
        maintain_aspect_ratio: true,
        target_width: 1920,
        target_height: 1920,
        expected_x: 1920.0,
        expected_y: 960.0,
      },
    ];

    for test_case in square_target_tall_source_with_padding_cases {
      // no padding
      let (x, y) = translate(
        test_case.x,
        test_case.y,
        test_case.source_width,
        test_case.source_height,
        test_case.target_width,
        test_case.target_height,
        test_case.maintain_aspect_ratio,
      );
      assert_eq!(
        x, test_case.expected_x,
        "square_target_tall_source_with_padding_cases: {test_case:?}"
      );
      assert_eq!(
        y, test_case.expected_y,
        "square_target_tall_source_with_padding_cases: {test_case:?}"
      );
    }
  }
}
