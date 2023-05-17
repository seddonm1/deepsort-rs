find . -name '.DS_Store' -type f -delete &&\
# python3 TrackEval/scripts/run_mot_challenge.py \
# --USE_PARALLEL True \
# --OUTPUT_DETAILED False \
# --PRINT_CONFIG False \
# --BENCHMARK MOT16 \
# --SPLIT_TO_EVAL train \
# --METRICS HOTA CLEAR Identity
python3 TrackEval/scripts/run_mot_challenge.py \
--USE_PARALLEL True \
--OUTPUT_DETAILED False \
--PRINT_CONFIG False \
--BENCHMARK MOT17 \
--SPLIT_TO_EVAL train \
--METRICS HOTA CLEAR Identity
python3 summary.py