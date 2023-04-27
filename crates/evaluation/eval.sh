rm -rf TrackEval/data/trackers/mot_challenge/MOT17-train/vctrack/data/MOT17-02-DPM.txt &&\
cargo run --release -- --input=TrackEval/data/gt/mot_challenge/MOT17-train/MOT17-02-DPM/img1/* &&\
python3 TrackEval/scripts/run_mot_challenge.py \
--USE_PARALLEL True \
--OUTPUT_DETAILED False \
--PRINT_CONFIG False \
--BENCHMARK MOT17 \
--METRICS HOTA CLEAR Identity