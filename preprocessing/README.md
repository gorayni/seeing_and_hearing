# Preprocessing

## Training dataset split

Splitting the action/verb/noun data for the challenge:

```bash
python preprocessing/create_annotations.py annotations --type action|verb|noun|baradel
```

## Audio

### Extracting the audio from all videos

Running FFMPEG Docker container:

```bash
docker pull jrottenberg/ffmpeg

docker run --rm -it --entrypoint='bash' \
      -v `realpath preprocessing/files_to_wav.sh`:/root/files_to_wav.sh  \
      -v `realpath preprocessing/to_wav.sh`:/root/to_wav.sh \
      -v `realpath EPIC_KITCHENS_2018`:/root/EPIC_KITCHENS_2018 \
      jrottenberg/ffmpeg
```
Once inside the Docker container

```bash
apt-get update && apt-get install parallel
cd ~ && ./to_wav.sh
```

### Extracting audio action recognition segments

Preparing the training and testing action segments

```bash
python preprocessing/extract_segments.py
```

Extracting audio segments

```bash
docker run --rm -it --entrypoint='bash' \
            -v `realpath annotations/action_segments.csv`:/root/action_segments.csv  \
            -v `realpath annotations/all_action_segments.csv`:/root/all_action_segments.csv  \
            -v `realpath annotations/s1_test_segments.csv`:/root/s1_test_segments.csv  \
            -v `realpath annotations/s2_test_segments.csv`:/root/s2_test_segments.csv  \
            -v `realpath preprocessing/to_action_audio_segments.sh`:/root/to_action_audio_segments.sh  \
            -v `realpath preprocessing/action_segment_to_wav.sh`:/root/action_segment_to_wav.sh \
            -v `realpath EPIC_KITCHENS_2018`:/root/EPIC_KITCHENS_2018 \
            jrottenberg/ffmpeg
```

Once inside the Docker container

```bash
apt-get update && apt-get install parallel
cd ~
./to_action_audio_segments.sh all_action_segments.csv train
./to_action_audio_segments.sh action_segments.csv train
./to_action_audio_segments.sh s1_test_segments.csv test
./to_action_audio_segments.sh s2_test_segments.csv test
```

#### Calculating audio spectrograms from segments

```bash
python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_train_action_labels.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/train' \
                                                    --hdf5_fname all_audio_dataset_4secs.hdf5

python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_ARC_train_labels.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/train' \
                                                    --hdf5_fname audio_dataset_4secs.hdf5
                                                    
python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_test_s1_timestamps.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/test' \
                                                    --hdf5_fname audio_test_s1_4secs.hdf5 \
                                                    --test
                                                    
python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_test_s2_timestamps.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/test' \
                                                    --hdf5_fname audio_test_s2_4secs.hdf5 \
                                                    --test
```
After obtaining the spectrograms, the statistics can be calculated as follows

```bash
python preprocessing/calculate_spectrograms_stats.py 'EPIC_KITCHENS_2018/action/audio/train/all_audio_dataset_4secs.hdf5' \
                                                     'EPIC_KITCHENS_2018/action/audio/train/all_spectrograms_stats.json'

python preprocessing/calculate_spectrograms_stats.py 'EPIC_KITCHENS_2018/action/audio/train/audio_dataset_4secs.hdf5' \
                                                     'EPIC_KITCHENS_2018/action/audio/train/spectrograms_stats.json'

python preprocessing/calculate_spectrograms_stats.py 'EPIC_KITCHENS_2018/action/audio/test/audio_test_s1_4secs.hdf5' \
                                                     'EPIC_KITCHENS_2018/action/audio/test/spectrograms_stats_test_s1.json'
                                                     
python preprocessing/calculate_spectrograms_stats.py 'EPIC_KITCHENS_2018/action/audio/test/audio_test_s2_4secs.hdf5' \
                                                     'EPIC_KITCHENS_2018/action/audio/test/spectrograms_stats_test_s2.json'
```

Calculating normalized spectrograms

```bash
python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_train_action_labels.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/train' \
                                                    --hdf5_fname all_audio_dataset_norm_4secs.hdf5 \
                                                    --norm_stats_json 'EPIC_KITCHENS_2018/action/audio/train/all_spectrograms_stats.json'

python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_ARC_train_labels.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/train' \
                                                    --hdf5_fname audio_dataset_norm_4secs.hdf5 \
                                                    --norm_stats_json 'EPIC_KITCHENS_2018/action/audio/train/spectrograms_stats.json'

python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_test_s1_timestamps.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/test' \
                                                    --hdf5_fname audio_test_s1_norm_4secs.hdf5 \
                                                    --test \
                                                    --norm_stats_json 'EPIC_KITCHENS_2018/action/audio/test/spectrograms_stats_test_s1.json'

python preprocessing/create_action_audio_pytable.py 'annotations/EPIC_test_s2_timestamps.pkl' \
                                                    'EPIC_KITCHENS_2018/action/audio/test' \
                                                    --hdf5_fname audio_test_s2_norm_4secs.hdf5 \
                                                    --test \
                                                    --norm_stats_json 'EPIC_KITCHENS_2018/action/audio/test/spectrograms_stats_test_s2.json'
```
## Creating Gulp files

Installing the epic-lib module

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install ./epic-lib
```

### Extracting action recognition segments

Running the action recognition extractor

```bash
./preprocessing/to_action_segments.sh
./preprocessing/to_test_action_segments.sh s1|s2
```
Creating Gulp files for RGB training

```bash
python3 -m epic_kitchens.gulp \
           EPIC_KITCHENS_2018/action/frames_rgb_flow/rgb/train \
           EPIC_KITCHENS_2018/action/gulp/rgb/train \
           annotations/EPIC_train_action_labels.pkl \
           rgb \
           --num-workers 26 \
           --segments-per-chunk 100
```

Creating Gulp files for optical flow training

```bash
python3 -m epic_kitchens.gulp \
           EPIC_KITCHENS_2018/action/frames_rgb_flow/flow/train \
           EPIC_KITCHENS_2018/action/gulp/flow/train \
           annotations/EPIC_train_action_labels.pkl \
           flow \
           --num-workers 26 \
           --segments-per-chunk 100
```

Creating Gulp files for RGB and optical flow for testing on S1

```bash
python3 -m epic_kitchens.gulp \
           EPIC_KITCHENS_2018/action/frames_rgb_flow/flow/test \
           EPIC_KITCHENS_2018/action/gulp/flow/test/s1 \
           annotations/EPIC_test_s1_timestamps.pkl \
           flow \
           --num-workers 12 \
           --segments-per-chunk 100 \
           --unlabelled

python3 -m epic_kitchens.gulp \
           EPIC_KITCHENS_2018/action/frames_rgb_flow/rgb/test \
           EPIC_KITCHENS_2018/action/gulp/rgb/test/s1 \
           annotations/EPIC_test_s1_timestamps.pkl rgb \
           --num-workers 12 \
           --segments-per-chunk 100 \
           --unlabelled
```
