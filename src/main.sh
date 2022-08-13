PYTHON=/home/xiaowei/miniconda3/envs/bci/bin/python

# config path
# CFG=/data/xiaowei/social_network/HENU/src/config/base_config.yaml
CFG=/data/xiaowei/social_network/HENU/src/config/test.yaml
# data path
EEG_DATA_PATH=/data/xiaowei/social_network/HENU/data/eeg_data
SURVEY_DATA_PATH=/data/xiaowei/social_network/HENU/data/survey_data

clear
$PYTHON /data/xiaowei/social_network/HENU/src/run.py --cfg=$CFG --eeg_data_path=$EEG_DATA_PATH --survey_data_path=$SURVEY_DATA_PATH

