find . -name '.DS_Store' -type f -delete &&\
cargo run --release -- generate \
--model=mars-small128.onnx \
--mot-dir=./TrackEval/data/gt/mot_challenge/MOT16-train \
--output-dir=./