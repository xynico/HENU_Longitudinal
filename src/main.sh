

# config path
CFG=/data/xiaowei/social_network/HENU/src/config/base_config.yaml

# data path
EEG_DATA_PATH=/data/xiaowei/social_network/HENU/data/eeg_data
BEHAVIOR_DATA_PATH=/data/xiaowei/social_network/HENU/data/behavior_data

$PYTHON /data/xiaowei/social_network/HENU/src/run.py --cfg=$CFG --eeg_data_path=$EEG_DATA_PATH --behavior_data_path=$BEHAVIOR_DATA_PATH

