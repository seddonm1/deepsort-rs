// mod utils;
use anyhow::{Ok, Result};
use clap::{Parser, Subcommand};
use deepsort_rs::{Detection, NearestNeighborDistanceMetric, Tracker};
use image::imageops;
use indexmap::IndexMap;
use rayon::prelude::*;
use std::{collections::HashMap, fs::OpenOptions, io::Write, path::PathBuf};
use tract_ndarray::{s, ShapeBuilder};
use tract_onnx::prelude::*;
// use utils::*;

use serde::Deserialize;

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
    /// Clones repos
    #[command(arg_required_else_help = true)]
    Generate {
        /// The path of the model to run
        #[arg(long)]
        model: PathBuf,

        /// The path of the mot groundtruth
        #[arg(long)]
        mot_dir: PathBuf,

        /// The path of the output files
        #[arg(long)]
        output_dir: PathBuf,
    },
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

        /// Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded.
        #[arg(long, default_value_t = 0)]
        min_detection_height: u32,

        /// Non-maxima suppression threshold: Maximum detection overlap.
        #[arg(long, default_value_t = 1.0)]
        nms_max_overlap: f32,

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
        Commands::Generate {
            model,
            mot_dir,
            output_dir: _,
        } => {
            let feature_session = tract_onnx::onnx()
                .model_for_path(model)?
                .with_output_fact(0, Default::default())?
                .into_optimized()?
                .into_runnable()?;

            let gts = glob::glob(&format!("{}/*/det/det.txt", mot_dir.to_string_lossy()))
                .unwrap()
                .filter_map(|path| path.ok())
                .collect::<Vec<_>>();

            for gt in gts {
                let sequence_base = gt.parent().unwrap().parent().unwrap();
                let image_base = sequence_base.join("img1");
                // let sequence = sequence_base
                //     .file_name()
                //     .unwrap()
                //     .to_str()
                //     .unwrap()
                //     .to_owned();

                let mut reader = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_path(gt.clone())?;
                let mot_detections = reader
                    .deserialize::<MotDetection>()
                    .map(|mot_detection| Ok(mot_detection?))
                    .collect::<Result<Vec<_>>>()?;

                let mut frame_mot_detections = HashMap::<usize, Vec<MotDetection>>::new();
                mot_detections.into_iter().for_each(|mot_detection| {
                    frame_mot_detections
                        .entry(mot_detection.frame)
                        .and_modify(|detections| detections.push(mot_detection.clone()))
                        .or_insert(vec![mot_detection]);
                });

                frame_mot_detections
                    .par_iter()
                    .map(|(frame, mot_detections)| {
                        let frame =
                            image::io::Reader::open(image_base.join(format!("{:0>6}.jpg", frame)))?
                                .decode()?
                                .to_rgb8();

                        mot_detections
                            .iter()
                            .map(|mot_detection| {
                                let detection = imageops::crop_imm(
                                    &frame,
                                    mot_detection.bb_left as u32,
                                    mot_detection.bb_top as u32,
                                    mot_detection.bb_width as u32,
                                    mot_detection.bb_top as u32,
                                )
                                .to_image();
                                let detection = imageops::resize(
                                    &detection,
                                    64,
                                    128,
                                    imageops::FilterType::Triangle,
                                );

                                let detection_tensor: Tensor =
                                    tract_ndarray::Array4::from_shape_fn(
                                        (1, 3, 128, 64),
                                        |(_, c, y, x)| detection[(x as _, y as _)][c] as f32,
                                    )
                                    .permuted_axes([0, 2, 3, 1])
                                    .into();

                                let results =
                                    feature_session.run(tvec!(detection_tensor.into()))?;
                                let results =
                                    results.first().unwrap().to_array_view::<f32>().unwrap();

                                println!("{:?}", results);

                                Ok(())
                            })
                            .collect::<Result<Vec<_>>>()?;

                        Ok(())
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
        }
        Commands::Evaluate {
            detection_dir,
            output_dir,
            min_confidence,
            min_detection_height: _,
            nms_max_overlap: _,
            max_cosine_distance,
            nn_budget,
        } => {
            let mut tracker = Tracker::default();
            tracker.with_nn_metric(
                NearestNeighborDistanceMetric::default()
                    .with_matching_threshold(max_cosine_distance)
                    .with_budget(nn_budget)
                    .to_owned(),
            );

            let npy_paths = glob::glob(&format!("{}/*.npy", detection_dir.to_string_lossy()))
                .unwrap()
                .filter_map(|path| path.ok())
                .collect::<Vec<_>>();

            for npy_path in npy_paths {
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
                let mut output_file =
                    OpenOptions::new()
                        .write(true)
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
                    })?;
            }
        }
    }

    // let font = Vec::from(include_bytes!("../DejaVuSans.ttf") as &[u8]);
    // let font = Font::try_from_vec(font).unwrap();

    // let mut tracker = Tracker::default();
    // tracker.with_nn_metric(
    //     NearestNeighborDistanceMetric::default()
    //         .with_budget(30)
    //         .to_owned(),
    // );

    // let feature_session = tract_onnx::onnx()
    //     .model_for_path("osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.onnx")?
    //     .with_output_fact(0, Default::default())?
    //     .into_optimized()?
    //     .into_runnable()?;

    // let files = glob::glob(&args.input)
    //     .unwrap()
    //     .filter_map(|path| path.ok())
    //     .collect::<Vec<_>>();

    // files
    //     .iter()
    //     .chunks(num_cpus::get())
    //     .into_iter()
    //     .try_for_each(|chunk| {
    //         let chunk_paths = chunk.collect::<Vec<_>>();
    //         let results = chunk_paths
    //             .par_iter()
    //             .flat_map(|path| {
    //                 let frame = image::io::Reader::open(path)
    //                     .unwrap()
    //                     .decode()
    //                     .unwrap()
    //                     .to_rgb8();

    //                 let resized = image_resize::resize_image(
    //                     &mut frame.to_vec(),
    //                     NonZeroU32::new(frame.width()).unwrap(),
    //                     NonZeroU32::new(frame.height()).unwrap(),
    //                     NonZeroU32::new(MODEL_WIDTH).unwrap(),
    //                     NonZeroU32::new(MODEL_HEIGHT).unwrap(),
    //                     true,
    //                     image_resize::Interpolation::Bilinear,
    //                 )
    //                 .unwrap();

    //                 let resized = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
    //                     MODEL_WIDTH,
    //                     MODEL_HEIGHT,
    //                     resized,
    //                 )
    //                 .unwrap();

    //                 let frame_tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
    //                     (1, 3, MODEL_HEIGHT as usize, MODEL_WIDTH as usize),
    //                     |(_, c, y, x)| resized[(x as _, y as _)][c] as f32,
    //                 )
    //                 .into();

    //                 let results = inference_session
    //                     .run(tvec!(
    //                         frame_tensor.into(),
    //                         max_output_boxes_per_class_tensor.clone().into(),
    //                         iou_threshold_tensor.clone().into(),
    //                         score_threshold_tensor.clone().into()
    //                     ))
    //                     .unwrap();

    //                 let results = results.first().unwrap().to_array_view::<f32>().unwrap();
    //                 results
    //                     .outer_iter()
    //                     .filter_map(|output| {
    //                         let class_index = output[5] as usize;
    //                         let confidence = output[6];
    //                         if class_index == 0 {
    //                             let (x0, y0) = frame_processing::translate(
    //                                 output[1],
    //                                 output[2],
    //                                 MODEL_WIDTH,
    //                                 MODEL_HEIGHT,
    //                                 frame.width(),
    //                                 frame.height(),
    //                                 true,
    //                             );
    //                             let (x1, y1) = frame_processing::translate(
    //                                 output[3],
    //                                 output[4],
    //                                 MODEL_WIDTH,
    //                                 MODEL_HEIGHT,
    //                                 frame.width(),
    //                                 frame.height(),
    //                                 true,
    //                             );

    //                             let x0 = x0.clamp(0.0, frame.width() as f32);
    //                             let y0 = y0.clamp(0.0, frame.height() as f32);
    //                             let x1 = x1.clamp(0.0, frame.width() as f32);
    //                             let y1 = y1.clamp(0.0, frame.height() as f32);

    //                             let feature_vector = if confidence < args.feature_threshold {
    //                                 None
    //                             } else {
    //                                 let person = imageops::crop_imm(
    //                                     &frame,
    //                                     x0 as u32,
    //                                     y0 as u32,
    //                                     (x1 - x0) as u32,
    //                                     (y1 - y0) as u32,
    //                                 )
    //                                 .to_image();
    //                                 let person = imageops::resize(
    //                                     &person,
    //                                     128,
    //                                     256,
    //                                     imageops::FilterType::Nearest,
    //                                 );

    //                                 let feature_tensor: Tensor =
    //                                     tract_ndarray::Array4::from_shape_fn(
    //                                         (1, 3, 256, 128),
    //                                         |(_, c, y, x)| person[(x as _, y as _)][c] as f32,
    //                                     )
    //                                     .into();

    //                                 let results =
    //                                     feature_session.run(tvec!(feature_tensor.into())).unwrap();

    //                                 Some(
    //                                     results
    //                                         .first()
    //                                         .unwrap()
    //                                         .to_array_view::<f32>()
    //                                         .unwrap()
    //                                         .as_slice()
    //                                         .unwrap()
    //                                         .to_vec(),
    //                                 )
    //                             };

    //                             Some((
    //                                 path.to_owned(),
    //                                 Detection::new(
    //                                     None,
    //                                     BoundingBox::new(x0, y0, x1 - x0, y1 - y0),
    //                                     confidence,
    //                                     None,
    //                                     None,
    //                                     feature_vector,
    //                                 ),
    //                             ))
    //                         } else {
    //                             None
    //                         }
    //                     })
    //                     .collect::<Vec<_>>()
    //             })
    //             .collect::<Vec<_>>();

    //         let mut frame_results = IndexMap::<PathBuf, Vec<Detection>>::new();
    //         results.into_iter().for_each(|(path, detection)| {
    //             frame_results
    //                 .entry(path.to_path_buf())
    //                 .and_modify(|detections| detections.push(detection.clone()))
    //                 .or_insert(vec![detection]);
    //         });

    //         frame_results.iter().try_for_each(|(path, detections)| {
    //             println!("{}", path.to_string_lossy());

    //             tracker.predict();
    //             tracker.update(detections)?;

    //             let frame_index = path.file_stem().unwrap().to_string_lossy().parse::<u32>()?;
    //             // write the tracker evaluation file
    //             let output_file = format!(
    //                 "TrackEval/data/trackers/mot_challenge/MOT17-train/{TRACKER_NAME}/data/{}.txt",
    //                 path.parent()
    //                     .unwrap()
    //                     .parent()
    //                     .unwrap()
    //                     .file_name()
    //                     .unwrap()
    //                     .to_string_lossy()
    //             );
    //             let output_path = Path::new(&output_file);
    //             std::fs::create_dir_all(output_path.parent().unwrap())?;
    //             let mut file = OpenOptions::new()
    //                 .append(true)
    //                 .create(true)
    //                 .open(output_file)?;

    //             let tracks = tracker
    //                 .tracks()
    //                 .iter()
    //                 .filter(|track| track.is_confirmed())
    //                 .collect::<Vec<_>>();

    //             tracks.iter().try_for_each(|track| {
    //                 // output format
    //                 // <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    //                 // 1,0.0,102.3751449584961,547.8447875976561,83.81769561767578,250.82427978515625,1,-1,-1,-1
    //                 file.write_all(
    //                     format!(
    //                         "{frame_index},{:.1},{:.3},{:.3},{:.3},{:.3},1,-1,-1,-1\n",
    //                         track.track_id() as f32,
    //                         track.bbox().x(),
    //                         track.bbox().y(),
    //                         track.bbox().width(),
    //                         track.bbox().height()
    //                     )
    //                     .as_bytes(),
    //                 )
    //             })?;

    //             if args.write_images {
    //                 write_images(path, tracks, &font)?
    //             };

    //             Ok(())
    //         })?;

    //         Ok(())
    //     })?;

    Ok(())
}

// fn write_images(path: &PathBuf, tracks: Vec<&Track>, font: &Font) -> Result<()> {
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
