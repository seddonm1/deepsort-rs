mod utils;
use anyhow::{Ok, Result};
use clap::{Parser, Subcommand};
use deepsort_rs::{BoundingBox, Detection, NearestNeighborDistanceMetric, Tracker};
use image::{imageops, ImageBuffer, Rgb};
use indexmap::IndexMap;
use npyz::WriterBuilder;
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    num::NonZeroU32,
    path::PathBuf,
};
use tract_itertools::Itertools;
use tract_ndarray::{s, Array, ShapeBuilder};
use tract_onnx::prelude::*;
use utils::{frame_processing, image_resize};

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
struct MotDetection {
    frame: usize,
    id: i32,
    bb_left: f32,
    bb_top: f32,
    bb_width: f32,
    bb_height: f32,
    conf: f32,
    x: i32,
    y: i32,
    z: i32,
}

/// A fictional versioning CLI
#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "track_eval")]
#[command(about = "A tester for tracking", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run object detection against the input files
    #[command(arg_required_else_help = true)]
    Detect {
        /// The path of the model to run
        #[arg(long)]
        model: PathBuf,

        /// The path of the mot groundtruth
        #[arg(long)]
        mot_dir: PathBuf,

        /// The maximum number of output boxes per class
        #[arg(short, long, default_value_t = NonZeroU32::new(50).unwrap())]
        max_output_boxes_per_class: NonZeroU32,

        /// The threshold for grouping detected objects
        #[arg(short, long, default_value_t = 0.5)]
        iou_threshold: f32,

        /// The score threshold for detection
        #[arg(short, long, default_value_t = 0.5)]
        score_threshold: f32,
    },
    /// Generate features and export to numpy
    #[command(arg_required_else_help = true)]
    Generate {
        /// The path of the model to run
        #[arg(long)]
        model: PathBuf,

        /// The path of the mot groundtruth
        #[arg(long)]
        mot_dir: PathBuf,

        /// The pattern to match for detection text files
        #[arg(long)]
        pattern: String,

        /// The path of the output files
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Load numpy, run tracking and produce MOT Challenge predictions
    #[command(arg_required_else_help = true)]
    Evaluate {
        /// Path to detections.
        #[arg(long)]
        detection_dir: PathBuf,

        /// Folder in which the results will be stored. Will be created if it does not exist.
        #[arg(long)]
        output_dir: PathBuf,

        /// Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
        #[arg(long, default_value_t = 0.0)]
        min_confidence: f32,

        /// Detection confidence threshold. Do not use any features lower than this confidence in tracking.
        #[arg(long, default_value_t = 0.0)]
        min_feature_confidence: f32,

        /// Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded.
        #[arg(long, default_value_t = 0)]
        min_detection_height: u32,

        /// Gating threshold for cosine distance metric (object appearance).
        #[arg(long, default_value_t = 1.0)]
        max_cosine_distance: f32,

        /// Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
        #[arg(long, default_value_t = 100)]
        nn_budget: usize,
    },
}

fn main() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Commands::Detect {
            model,
            mot_dir,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        } => {
            let session = tract_onnx::onnx()
                .model_for_path(model)?
                .with_output_fact(0, Default::default())?
                .into_optimized()?
                .into_runnable()?;

            let input_shapes = session
                .model
                .input_fact(0)?
                .shape
                .iter()
                .map(|dim| Ok(dim.to_i64()?))
                .collect::<Result<Vec<_>>>()?;
            let input_width = *input_shapes.get(3).unwrap() as usize;
            let input_height = *input_shapes.get(2).unwrap() as usize;

            let max_output_boxes_per_class_tensor: Tensor =
                Array::from_elem(1, max_output_boxes_per_class.get() as i64)
                    .into_shape(vec![1])?
                    .into();
            let iou_threshold_tensor: Tensor = Array::from_elem(1, iou_threshold)
                .into_shape(vec![1])?
                .into();
            let score_threshold_tensor: Tensor = Array::from_elem(1, score_threshold)
                .into_shape(vec![1])?
                .into();

            let sequence_img_paths: Vec<PathBuf> =
                glob::glob(&format!("{}/*/img1", mot_dir.to_string_lossy()))
                    .unwrap()
                    .filter_map(|path| path.ok())
                    .collect::<Vec<_>>();

            sequence_img_paths
                .iter()
                .try_for_each(|sequence_img_path| {
                    println!("{}", sequence_img_path.parent().unwrap().to_string_lossy());

                    let imgs =
                        glob::glob(&format!("{}/*.jpg", sequence_img_path.to_string_lossy()))
                            .unwrap()
                            .filter_map(|path| path.ok())
                            .collect::<Vec<_>>();

                    let detections = imgs
                        .into_iter()
                        .chunks(num_cpus::get())
                        .into_iter()
                        .map(|chunk| {
                            let chunk_paths = chunk.collect::<Vec<_>>();
                            Ok(chunk_paths
                                .par_iter()
                                .flat_map(|path| {
                                    let frame = image::io::Reader::open(path)
                                        .unwrap()
                                        .decode()
                                        .unwrap()
                                        .to_rgb8();

                                    let resized = image_resize::resize_image(
                                        &mut frame.to_vec(),
                                        NonZeroU32::new(frame.width()).unwrap(),
                                        NonZeroU32::new(frame.height()).unwrap(),
                                        NonZeroU32::new(input_width as u32).unwrap(),
                                        NonZeroU32::new(input_height as u32).unwrap(),
                                        true,
                                        image_resize::Interpolation::Bilinear,
                                    )
                                    .unwrap();

                                    let resized = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
                                        input_width as u32,
                                        input_height as u32,
                                        resized,
                                    )
                                    .unwrap();

                                    let frame_tensor: Tensor =
                                        tract_ndarray::Array4::from_shape_fn(
                                            (1, 3, input_height, input_width),
                                            |(_, c, y, x)| resized[(x as _, y as _)][c] as f32,
                                        )
                                        .into();

                                    let results = session
                                        .run(tvec!(
                                            frame_tensor.into(),
                                            max_output_boxes_per_class_tensor.clone().into(),
                                            iou_threshold_tensor.clone().into(),
                                            score_threshold_tensor.clone().into()
                                        ))
                                        .unwrap();

                                    let results =
                                        results.first().unwrap().to_array_view::<f32>().unwrap();
                                    results
                                        .outer_iter()
                                        .filter_map(|output| {
                                            let class_index = output[5] as usize;
                                            let confidence = output[6];
                                            if class_index == 0 {
                                                let (x0, y0) = frame_processing::translate(
                                                    output[1],
                                                    output[2],
                                                    input_width as u32,
                                                    input_height as u32,
                                                    frame.width(),
                                                    frame.height(),
                                                    true,
                                                );
                                                let (x1, y1) = frame_processing::translate(
                                                    output[3],
                                                    output[4],
                                                    input_width as u32,
                                                    input_height as u32,
                                                    frame.width(),
                                                    frame.height(),
                                                    true,
                                                );

                                                let x0 = x0.clamp(0.0, frame.width() as f32);
                                                let y0 = y0.clamp(0.0, frame.height() as f32);
                                                let x1 = x1.clamp(0.0, frame.width() as f32);
                                                let y1 = y1.clamp(0.0, frame.height() as f32);

                                                Some((
                                                    path.to_owned(),
                                                    Detection::new(
                                                        None,
                                                        BoundingBox::new(x0, y0, x1 - x0, y1 - y0),
                                                        confidence,
                                                        None,
                                                        None,
                                                        None,
                                                    ),
                                                ))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>())
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>();

                    let output_dir = mot_dir
                        .join(
                            &*sequence_img_path
                                .parent()
                                .unwrap()
                                .file_name()
                                .unwrap()
                                .to_string_lossy(),
                        )
                        .join("det");
                    std::fs::create_dir_all(&output_dir)?;
                    let mut output_file = OpenOptions::new()
                        .write(true)
                        .truncate(true)
                        .create(true)
                        .open(output_dir.join("vc_det.txt"))?;

                    detections.iter().try_for_each(|(path, detection)| {
                        // output format
                        // <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                        // 1,0.0,102.3751449584961,547.8447875976561,83.81769561767578,250.82427978515625,1,-1,-1,-1
                        output_file.write_all(
                            format!(
                                "{},-1,{:.3},{:.3},{:.3},{:.3},{:.3},-1,-1,-1\n",
                                path.file_stem()
                                    .unwrap()
                                    .to_string_lossy()
                                    .parse::<usize>()?,
                                detection.bbox().x(),
                                detection.bbox().y(),
                                detection.bbox().width(),
                                detection.bbox().height(),
                                detection.confidence()
                            )
                            .as_bytes(),
                        )?;

                        Ok(())
                    })?;

                    Ok(())
                })?;
        }
        Commands::Generate {
            model,
            mot_dir,
            pattern,
            output_dir,
        } => {
            let session = tract_onnx::onnx()
                .model_for_path(model)?
                .with_output_fact(0, Default::default())?
                .into_optimized()?
                .into_runnable()?;

            let input_shapes = session
                .model
                .input_fact(0)?
                .shape
                .iter()
                .map(|dim| Ok(dim.to_i64()?))
                .collect::<Result<Vec<_>>>()?;
            let input_width = *input_shapes.get(3).unwrap() as usize;
            let input_height = *input_shapes.get(2).unwrap() as usize;

            let output_shapes = session
                .model
                .output_fact(0)?
                .shape
                .iter()
                .map(|dim| Ok(dim.to_i64()?))
                .collect::<Result<Vec<_>>>()?;
            let feature_length = *output_shapes.get(1).unwrap() as usize;

            let dets = glob::glob(&format!("{}/*/det/{}", mot_dir.to_string_lossy(), pattern))
                .unwrap()
                .filter_map(|path| path.ok())
                .collect::<Vec<_>>();

            dets.iter().try_for_each(|det| {
                println!("{:?}", det);
                let sequence_base = det.parent().unwrap().parent().unwrap();
                let image_base = sequence_base.join("img1");
                let sequence = sequence_base
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned();

                let mut reader = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_path(det.clone())?;
                let mot_detections = reader
                    .deserialize::<MotDetection>()
                    .map(|mot_detection| Ok(mot_detection?))
                    .collect::<Result<Vec<_>>>()?;

                let num_detections = mot_detections.len();

                let mut frame_mot_detections = IndexMap::<usize, Vec<MotDetection>>::new();
                mot_detections.into_iter().for_each(|mot_detection| {
                    frame_mot_detections
                        .entry(mot_detection.frame)
                        .and_modify(|detections| detections.push(mot_detection.clone()))
                        .or_insert(vec![mot_detection]);
                });

                let data = frame_mot_detections
                    .iter()
                    .flat_map(|(frame_index, mot_detections)| {
                        println!("{}: {}", sequence, frame_index);

                        let frame = image::io::Reader::open(
                            image_base.join(format!("{:0>6}.jpg", frame_index)),
                        )?
                        .decode()?
                        .to_rgb8();

                        Ok(mot_detections
                            .par_iter()
                            .map(|mot_detection| {
                                let detection = imageops::crop_imm(
                                    &frame,
                                    mot_detection.bb_left as u32,
                                    mot_detection.bb_top as u32,
                                    mot_detection.bb_width as u32,
                                    mot_detection.bb_height as u32,
                                )
                                .to_image();

                                let detection = imageops::resize(
                                    &detection,
                                    input_width as u32,
                                    input_height as u32,
                                    imageops::FilterType::Triangle,
                                );

                                let detection_tensor: Tensor =
                                    tract_ndarray::Array4::from_shape_fn(
                                        (1, 3, input_height, input_width),
                                        |(_, c, y, x)| {
                                            let mean = [0.485, 0.456, 0.406][c];
                                            let std = [0.229, 0.224, 0.225][c];
                                            (detection[(x as _, y as _)][c] as f32 / 255.0 - mean)
                                                / std
                                        },
                                    )
                                    .into();

                                let results = session.run(tvec!(detection_tensor.into()))?;

                                let base = vec![
                                    *frame_index as f64,
                                    mot_detection.id as f64,
                                    mot_detection.bb_left as f64,
                                    mot_detection.bb_top as f64,
                                    mot_detection.bb_width as f64,
                                    mot_detection.bb_height as f64,
                                    mot_detection.conf as f64,
                                    mot_detection.x as f64,
                                    mot_detection.y as f64,
                                    mot_detection.z as f64,
                                ];

                                let feature_vector = results
                                    .first()
                                    .unwrap()
                                    .to_array_view::<f32>()
                                    .unwrap()
                                    .as_standard_layout()
                                    .as_slice()
                                    .unwrap()
                                    .iter()
                                    .map(|v| *v as f64)
                                    .collect::<Vec<_>>();

                                Ok([base, feature_vector].concat())
                            })
                            .collect::<Result<Vec<_>>>()?)
                    })
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>();

                let file = std::io::BufWriter::new(File::create(
                    output_dir.join(format!("./{sequence}.npy")),
                )?);
                let mut writer = npyz::WriteOptions::<f64>::new()
                    .default_dtype()
                    .shape(&[num_detections as u64, 10 + feature_length as u64])
                    .writer(file)
                    .begin_nd()?;
                writer.extend(data)?;
                Ok(writer.finish()?)
            })?;
        }
        Commands::Evaluate {
            detection_dir,
            output_dir,
            min_confidence,
            min_feature_confidence,
            min_detection_height: _,
            max_cosine_distance,
            nn_budget,
        } => {
            let npy_paths = glob::glob(&format!("{}/*.npy", detection_dir.to_string_lossy()))
                .unwrap()
                .filter_map(|path| path.ok())
                .collect::<Vec<_>>();

            npy_paths.iter().try_for_each(|npy_path| {
                println!("{:?}", npy_path);
                let mut tracker = Tracker::default();
                tracker.with_nn_metric(
                    NearestNeighborDistanceMetric::default()
                        .with_matching_threshold(max_cosine_distance)
                        .with_budget(nn_budget)
                        .to_owned(),
                );

                let bytes = std::fs::read(npy_path.clone())?;
                let npy = npyz::NpyFile::new(&bytes[..])?;
                let shape = npy.shape().to_owned();
                let order = npy.order();

                let data = npy
                    .data::<f64>()?
                    .map(|data| Ok(data?))
                    .collect::<Result<Vec<f64>>>()?;

                let shape = match shape[..] {
                    [i1, i2] => [i1 as usize, i2 as usize],
                    _ => panic!("expected 2D array"),
                };
                let true_shape = shape.set_f(order == npyz::Order::Fortran);
                let array = ndarray::Array2::from_shape_vec(true_shape, data)?;

                let detections = array
                    .outer_iter()
                    .map(|item| {
                        (
                            item[0] as usize,
                            deepsort_rs::Detection::new(
                                None,
                                deepsort_rs::BoundingBox::new(
                                    item[2] as f32,
                                    item[3] as f32,
                                    item[4] as f32,
                                    item[5] as f32,
                                ),
                                item[6] as f32,
                                Some(1),
                                Some("person".to_string()),
                                Some(item.slice(s![10..]).map(|v| *v as f32).to_vec()),
                            ),
                        )
                    })
                    .collect::<Vec<_>>();

                let mut frame_detections = IndexMap::<usize, Vec<Detection>>::new();
                detections.into_iter().for_each(|(frame_index, detection)| {
                    frame_detections
                        .entry(frame_index)
                        .and_modify(|detections| detections.push(detection.clone()))
                        .or_insert(vec![detection]);
                });

                std::fs::create_dir_all(output_dir.clone())?;
                let mut output_file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open(output_dir.join(format!(
                        "{}.txt",
                        npy_path.file_stem().unwrap().to_string_lossy()
                    )))?;

                frame_detections
                    .into_iter()
                    .try_for_each(|(frame_index, detections)| {
                        println!("Procesing frame {:0>4}", frame_index);

                        let detections = detections
                            .into_iter()
                            .filter(|detection| detection.confidence() > min_confidence)
                            .map(|mut detection| {
                                // remove feature vector if below threshold (will become iou/sort tracking only)
                                if detection.confidence() < min_feature_confidence {
                                    *detection.feature_mut() = None;
                                };

                                detection
                            })
                            .collect::<Vec<_>>();

                        tracker.predict();
                        tracker.update(&detections)?;

                        tracker
                            .tracks()
                            .iter()
                            .filter(|track| track.is_confirmed() && track.time_since_update() <= 1)
                            .try_for_each(|track| {
                                output_file.write_all(
                                    format!(
                                        "{frame_index},{},{:.2},{:.2},{:.2},{:.2},1,-1,-1,-1\n",
                                        track.track_id(),
                                        track.bbox().x(),
                                        track.bbox().y(),
                                        track.bbox().width(),
                                        track.bbox().height()
                                    )
                                    .as_bytes(),
                                )
                            })?;

                        Ok(())
                    })
            })?;
        }
    }

    Ok(())
}

// fn write_images(path: &PathBuf, tracks: Vec<&Track>, font: &Font) -> Result<()> {

// let font = Vec::from(include_bytes!("../DejaVuSans.ttf") as &[u8]);
// let font = Font::try_from_vec(font).unwrap();
//     let mut frame = image::io::Reader::open(path)
//         .unwrap()
//         .decode()
//         .unwrap()
//         .to_rgb8();

//     tracks.iter().for_each(|track| {
//         let color = match track.match_source() {
//             Some(match_source) => match match_source {
//                 MatchSource::NearestNeighbor { .. } => Rgb([255u8, 0u8, 0u8]),
//                 MatchSource::IoU { .. } => Rgb([0u8, 0u8, 255u8]),
//             },
//             None => Rgb([255u8, 255u8, 255u8]),
//         };

//         imageproc::drawing::draw_hollow_rect_mut(
//             &mut frame,
//             Rect::at(
//                 track.detection().bbox().x() as i32,
//                 track.detection().bbox().y() as i32,
//             )
//             .of_size(
//                 track.detection().bbox().width() as u32,
//                 track.detection().bbox().height() as u32,
//             ),
//             color,
//         );

//         imageproc::drawing::draw_text_mut(
//             &mut frame,
//             color,
//             (track.detection().bbox().x() as i32) + 5,
//             (track.detection().bbox().y() as i32) + 5,
//             Scale { x: 25.0, y: 25.0 },
//             font,
//             &track.track_id().to_string(),
//         );
//     });

//     let file_name = format!("./out/{}", path.file_name().unwrap().to_string_lossy());
//     Ok(frame.save(file_name)?)
// }
