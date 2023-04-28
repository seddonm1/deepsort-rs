rm -rf TrackEval/data/trackers/mot_challenge/MOT17-train/vctrack/data/MOT17-04-FRCNN.txt &&\
find . -name '.DS_Store' -type f -delete &&\
cargo run --release -- \
--input=TrackEval/data/gt/mot_challenge/MOT17-train/MOT17-04-FRCNN/img1/* \
--max-output-boxes-per-class=50 \
--iou-threshold=0.5 \
--score-threshold=0.5 \
--feature-threshold=0.5 \
--write-images