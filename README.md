[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15293194.svg)](https://doi.org/10.5281/zenodo.15293194)

# Interspeech 2025 - Speech Emotion Recognition challenge

UNICAMP entry for the 2025 Interspeech challenge on speech emotion recognition.

In order to run training you must have at least one 30gb GPU, otherwise you should decrease batch size but it was not tested.

## Building environment
Take this steps to prepare data and environment.

### Data preparation

Create 'data' dir and copy challenge data in it (must be 'data' because it is in gitignore to avoid pushing):
```bash
mkdir data
```
Copy all challenge files to this folder
```bash
cp -r /home/$USER/IS2025_Podcast_Challenge/* ./data/
```
Extract zip files and tar.gz files
```bash
unzip Transcripts.zip
unzip Labels.zip
tar -xzvf Audios.tar.gz
```
After all that you should see this on your data folder:
![image](https://github.com/user-attachments/assets/65afb13d-fba4-423f-bc21-bf66d69b756d)

### Docker environment setup

In order to generate the docker image we took the same requirements as in benchmark provided by challenge organizers which depends on Python 3.9.6+, CUDA 11.6 and pytorch 1.13.1

Build docker image with 'ser' tag (standing for Speech Emotion Recognition)
```bash
docker build -t ser .
```
Running docker container
```bash
sh docker-run.sh -g 3 -n 3 -p "1213:1213"
```
This should initiate your container in GPU id 3, with name "ser3" exported to port 1213

To access remotely (like jupyter) run this in your local machine terminal
```bash
ssh -N -L 1213:localhost:1213 dl-17
```
Make sure to export the same port as the running container and the correct dl

## Experiment running
Take this steps for experiment running.

### 1) Whisper transcriptions
There is a whisper_transcript.csv file already. The code used to generate it is in notebook: ./test/Whisper Transcription.ipynb.

### 2) Computing SSL embeddings

For roberta embeddings, run:
```bash
python preprocessing/preprocess_roberta.py --roberta_type FacebookAI/roberta-large --df_path ./test/whisper_transcript.csv --save_path data_tmp/roberta_large
```

For deberta embeddings, run:
```bash
python preprocessing/preprocess_deroberta.py --roberta_type microsoft/deberta-v2-xlarge --df_path ./test/whisper_transcript.csv --save_path data_tmp/deberta_xxlarge_v2
```

For speech SSL models, run:
```bash
python preprocessing/preprocess_speech.py --ssl_type $SSL_TYPE --wav_dir $WAVS_DIR --save_path data_tmp/$SSL_NAME
```

SSL_TYPE ∈ {`microsoft/wavlm-large`, `facebook/wav2vec2-xls-r-2b`, `facebook/hubert-xlarge-ls960-ft`}

For whisper model, run:
```bash
python preprocessing/preprocess_whisper.py --ssl_type openai/whisper-large-v3 --wav_dir $WAVS_DIR --save_path data_tmp/whisper-large-v3
```


For NS3 embeddings, first get the encoder and decoder v2 from https://huggingface.co/amphion/naturalspeech3_facodec/tree/main:

```bash
mkdir pretrained_models
cd pretrained_model

wget https://huggingface.co/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder_v2.bin?download=true
wget https://huggingface.co/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder_v2.bin?download=true
```
Then get the embeddings running:

```bash
python preprocessing/preprocess_ns3_prosody.py --wav_dir $WAVS_DIR --save_path data_tmp/ns3_prosody_emb
```

### 3) Training

Please take a look on the ./configs/ folder to get all configs used. All training scripts run regarding the config file. Once you have a configuration you can run the training with:

For not-balanced training:
```bash
python preprocessing/train_cat_bimodal_lazy_1head.py --config_path $CONFIG_PATH
```

For balanced training:
```bash
python preprocessing/train_cat_bimodal_lazy_1head_ranking.py --config_path $CONFIG_PATH
```

For not-balanced training TRI MODAL:
```bash
python preprocessing/train_cat_trimodal_lazy_1head.py --config_path $CONFIG_PATH
```

For balanced training TRI MODAL:
```bash
python preprocessing/train_cat_trimodal_lazy_1head_ranking.py --config_path $CONFIG_PATH
```

### Evaluation and testing
Eval and testing are done by generating probas from the scripts files:

```bash
python preprocessing/eval_cat_bimodal_lazy_1head.py --config_path $CONFIG_PATH
python preprocessing/eval_cat_bimodal_lazy_1head_ranking.py --config_path $CONFIG_PATH
...
```

Also, extract the probas for training data as well:

```bash
python preprocessing/extract_train_cat_bimodal_lazy_1head.py --config_path $CONFIG_PATH --train_df $TRAIN_DF
python preprocessing/extract_train_cat_trimodal_lazy_1head.py --config_path $CONFIG_PATH --train_df $TRAIN_DF
...
```

$TRAIN_DF just need to have a "wav_path" column with the wavs you want to calculate to avoid calculating for the entire train dataset. We used the "./test/train_stacking_sample.csv" wavs.

To effectively evaluate and generate submission files please rely on "./test/" EVAL and test jupyter notebooks. The 5-fold RandomForest models used for the final submission are on release files.
