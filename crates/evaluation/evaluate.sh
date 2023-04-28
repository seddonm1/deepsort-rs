find . -name '.DS_Store' -type f -delete &&\
python3 TrackEval/scripts/run_mot_challenge.py \
--USE_PARALLEL True \
--OUTPUT_DETAILED False \
--PRINT_CONFIG False \
--BENCHMARK MOT17 \
--METRICS HOTA CLEAR Identity