find . -name '.DS_Store' -type f -delete &&\
cargo build --release &&\
# time cargo run --release -- evaluate \
# --detection-dir=./resources/detections/MOT16-train/*/*.npy \
# --output-dir=TrackEval/data/trackers/mot_challenge/MOT16-train/vctrack/data \
# --min-confidence=0.3 \
# --min-feature-confidence=1.0 \
# --min-detection-height=0 \
# --max-cosine-distance=0.2 \
# --nn-budget=100
time cargo run --release -- evaluate \
--detection-dir=TrackEval/data/gt/mot_challenge/MOT17-train/MOT17-02-DPM/det/bytetrack.txt \
--output-dir=TrackEval/data/trackers/mot_challenge/MOT17-train/vctrack/data