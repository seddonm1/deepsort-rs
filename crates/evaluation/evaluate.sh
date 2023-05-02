find . -name '.DS_Store' -type f -delete &&\
cargo build --release &&\
time cargo run --release -- evaluate \
--detection-dir=deep_sort/resources/detections/MOT16-train \
--output-dir=TrackEval/data/trackers/mot_challenge/MOT16-train/vctrack/data \
--min-confidence=-100.0 \
--min-detection-height=0 \
--nms-max-overlap=0.5 \
--max-cosine-distance=0.2 \
--nn-budget=100