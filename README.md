# Interspeech 2025 - Speech Emotion Recognition challenge

UNICAMP entry for the 2025 Interspeech challenge on speech emotion recognition.

In order to run training you must have at least one 30gb GPU, otherwise you should decrease batch size but it was not tested.

## Building environment
Take this steps to prepare data and environment.

#### 1) Data preparing

Create 'data' dir and copy challenge data in it (must be 'data' because it is in gitignore to avoid pushing):
```bash
mkdir data
```
Copy all challenge files to this folder
```bash
cp -r /home/lucas.ueda/IS2025_Podcast_Challenge/* ./data/
```
Extract zip files and tar.gz files
```bash
unzip Transcripts.zip
unzip Labels.zip
tar -xzvf Audios.tar.gz
```
After all that you should see this on your data folder:
![image](https://github.com/user-attachments/assets/65afb13d-fba4-423f-bc21-bf66d69b756d)

#### 2) Docker environment setup

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
