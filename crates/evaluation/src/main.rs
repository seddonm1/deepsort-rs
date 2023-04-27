mod utils;
use anyhow::{Ok, Result};
use clap::Parser;
use deepsort_rs::{BoundingBox, Detection, MatchSource, NearestNeighborDistanceMetric, Tracker};
use image::{imageops, ImageBuffer, Rgb};
use imageproc::rect::Rect;
use indexmap::IndexMap;
use itertools::*;
use rayon::prelude::*;
use rusttype::{Font, Scale};
use std::{
    fs::OpenOptions,
    io::Write,
    num::NonZeroU32,
    path::{Path, PathBuf},
};
use tract_ndarray::Array;
use tract_onnx::prelude::*;
use utils::*;

static MODEL_WIDTH: u32 = 640;
static MODEL_HEIGHT: u32 = 640;
static TRACKER_NAME: &str = "vctrack";

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input path
    #[arg(short, long)]
    input: String,

    /// The maximum number of output boxes per class
    #[arg(short, long, default_value_t = NonZeroU32::new(50).unwrap())]
    max_output_boxes_per_class: NonZeroU32,

    /// The threshold for grouping detected objects
    #[arg(short, long, default_value_t = 0.5)]
    iou_threshold: f32,

    /// The score threshold for detection
    #[arg(short, long, default_value_t = 0.5)]
    score_threshold: f32,

    /// The minimum confidence required to perform feature matching
    #[arg(short, long, default_value_t = 0.6)]
    feature_threshold: f32,

    /// Write debug images
    #[arg(short, long, default_value_t = false)]
    write_images: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let font = Vec::from(include_bytes!("../DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();

    let mut tracker = Tracker::default();
    tracker.with_nn_metric(
        NearestNeighborDistanceMetric::default()
            .with_budget(30)
            .to_owned(),
    );

    let inference_session = tract_onnx::onnx()
        .model_for_path("yolov8m_onms_fp32.onnx")?
        .with_output_fact(0, Default::default())?
        .into_optimized()?
        .into_runnable()?;

    let max_output_boxes_per_class_tensor: Tensor =
        Array::from_elem(1, args.max_output_boxes_per_class.get() as i64)
            .into_shape(vec![1])?
            .into();
    let iou_threshold_tensor: Tensor = Array::from_elem(1, args.iou_threshold)
        .into_shape(vec![1])?
        .into();
    let score_threshold_tensor: Tensor = Array::from_elem(1, args.score_threshold)
        .into_shape(vec![1])?
        .into();

    let feature_session = tract_onnx::onnx()
        .model_for_path("osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.onnx")?
        .with_output_fact(0, Default::default())?
        .into_optimized()?
        .into_runnable()?;

    let files = glob::glob(&args.input)
        .unwrap()
        .filter_map(|path| path.ok())
        .collect::<Vec<_>>();

    files
        .iter()
        .chunks(num_cpus::get())
        .into_iter()
        .try_for_each(|chunk| {
            let chunk_paths = chunk.collect::<Vec<_>>();
            let results = chunk_paths
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
                        NonZeroU32::new(MODEL_WIDTH).unwrap(),
                        NonZeroU32::new(MODEL_HEIGHT).unwrap(),
                        true,
                        image_resize::Interpolation::Bilinear,
                    )
                    .unwrap();

                    let resized = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
                        MODEL_WIDTH,
                        MODEL_HEIGHT,
                        resized,
                    )
                    .unwrap();

                    let frame_tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
                        (1, 3, MODEL_HEIGHT as usize, MODEL_WIDTH as usize),
                        |(_, c, y, x)| resized[(x as _, y as _)][c] as f32,
                    )
                    .into();

                    let results = inference_session
                        .run(tvec!(
                            frame_tensor.into(),
                            max_output_boxes_per_class_tensor.clone().into(),
                            iou_threshold_tensor.clone().into(),
                            score_threshold_tensor.clone().into()
                        ))
                        .unwrap();

                    let results = results.first().unwrap().to_array_view::<f32>().unwrap();
                    results
                        .outer_iter()
                        .filter_map(|output| {
                            let class_index = output[5] as usize;
                            let confidence = output[6];
                            if class_index == 0 {
                                let (x0, y0) = frame_processing::translate(
                                    output[1],
                                    output[2],
                                    MODEL_WIDTH,
                                    MODEL_HEIGHT,
                                    frame.width(),
                                    frame.height(),
                                    true,
                                );
                                let (x1, y1) = frame_processing::translate(
                                    output[3],
                                    output[4],
                                    MODEL_WIDTH,
                                    MODEL_HEIGHT,
                                    frame.width(),
                                    frame.height(),
                                    true,
                                );

                                let x0 = x0.clamp(0.0, frame.width() as f32);
                                let y0 = y0.clamp(0.0, frame.height() as f32);
                                let x1 = x1.clamp(0.0, frame.width() as f32);
                                let y1 = y1.clamp(0.0, frame.height() as f32);

                                let feature_vector = if confidence < args.feature_threshold {
                                    None
                                } else {
                                    let person = imageops::crop_imm(
                                        &frame,
                                        x0 as u32,
                                        y0 as u32,
                                        (x1 - x0) as u32,
                                        (y1 - y0) as u32,
                                    )
                                    .to_image();
                                    let person = imageops::resize(
                                        &person,
                                        128,
                                        256,
                                        imageops::FilterType::Nearest,
                                    );

                                    let feature_tensor: Tensor =
                                        tract_ndarray::Array4::from_shape_fn(
                                            (1, 3, 256, 128),
                                            |(_, c, y, x)| person[(x as _, y as _)][c] as f32,
                                        )
                                        .into();

                                    let results =
                                        feature_session.run(tvec!(feature_tensor.into())).unwrap();

                                    Some(
                                        results
                                            .first()
                                            .unwrap()
                                            .to_array_view::<f32>()
                                            .unwrap()
                                            .as_slice()
                                            .unwrap()
                                            .to_vec(),
                                    )
                                };

                                Some((
                                    path.to_owned(),
                                    Detection::new(
                                        None,
                                        BoundingBox::new(x0, y0, x1 - x0, y1 - y0),
                                        confidence,
                                        None,
                                        None,
                                        feature_vector,
                                    ),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut frame_results = IndexMap::<PathBuf, Vec<Detection>>::new();
            results.into_iter().for_each(|(path, detection)| {
                frame_results
                    .entry(path.to_path_buf())
                    .and_modify(|detections| detections.push(detection.clone()))
                    .or_insert(vec![detection]);
            });

            frame_results.iter().try_for_each(|(path, detections)| {
                println!("{}", path.to_string_lossy());

                tracker.predict();
                tracker.update(detections)?;

                let frame_index = path.file_stem().unwrap().to_string_lossy().parse::<u32>()?;
                // write the tracker evaluation file
                let output_file = format!(
                    "TrackEval/data/trackers/mot_challenge/MOT17-train/{TRACKER_NAME}/data/{}.txt",
                    path.parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                );
                let output_path = Path::new(&output_file);
                std::fs::create_dir_all(output_path.parent().unwrap())?;
                let mut file = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(output_file)?;

                tracker.tracks().iter().try_for_each(|track| {
                    // output format
                    // <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    // 1,0.0,102.3751449584961,547.8447875976561,83.81769561767578,250.82427978515625,1,-1,-1,-1
                    file.write_all(
                        format!(
                            "{frame_index},{:.1},{:.3},{:.3},{:.3},{:.3},1,-1,-1,-1\n",
                            track.track_id() as f32,
                            track.bbox().x(),
                            track.bbox().y(),
                            track.bbox().width(),
                            track.bbox().height()
                        )
                        .as_bytes(),
                    )
                })?;

                if args.write_images {
                    write_images(path, &tracker, &font)?
                };

                Ok(())
            })?;

            Ok(())
        })?;

    Ok(())
}

fn write_images(path: &PathBuf, tracker: &Tracker, font: &Font) -> Result<()> {
    let mut frame = image::io::Reader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    tracker.tracks().iter().for_each(|track| {
        let color = match track.match_source() {
            Some(match_source) => match match_source {
                MatchSource::NearestNeighbor { .. } => Rgb([255u8, 0u8, 0u8]),
                MatchSource::IoU { .. } => Rgb([0u8, 0u8, 255u8]),
            },
            None => Rgb([255u8, 255u8, 255u8]),
        };

        imageproc::drawing::draw_hollow_rect_mut(
            &mut frame,
            Rect::at(
                track.detection().bbox().x() as i32,
                track.detection().bbox().y() as i32,
            )
            .of_size(
                track.detection().bbox().width() as u32,
                track.detection().bbox().height() as u32,
            ),
            color,
        );

        imageproc::drawing::draw_text_mut(
            &mut frame,
            color,
            (track.detection().bbox().x() as i32) + 5,
            (track.detection().bbox().y() as i32) + 5,
            Scale { x: 25.0, y: 25.0 },
            font,
            &track.track_id().to_string(),
        );
    });

    let file_name = format!("./out/{}", path.file_name().unwrap().to_string_lossy());
    Ok(frame.save(file_name)?)
}
