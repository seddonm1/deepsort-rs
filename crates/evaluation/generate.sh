find . -name '.DS_Store' -type f -delete &&\
cargo run --release -- generate \
--model=osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.onnx \
--mot-dir=./TrackEval/data/gt/mot_challenge/MOT16-train \
--output-dir=./resources/detections/MOT16-train