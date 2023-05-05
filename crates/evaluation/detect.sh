find . -name '.DS_Store' -type f -delete &&\
cargo run --release -- detect \
--model=yolov8m_onms_fp32.onnx \
--mot-dir=./TrackEval/data/gt/mot_challenge/MOT16-train