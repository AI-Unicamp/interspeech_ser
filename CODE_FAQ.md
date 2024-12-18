# Local Code FAQ

Just some general questions about the code, that can be (or not) important.

### How dataloader from benchmark works?
> It works by loading all waveforms in memory, and then passing to 3 datasets: wavs, labels and utterance. The output of dataloader in each iteration is (respectively): waveform, label, waveform_mask, utterance_id. Utterance_Id is basically a string which can be retrieved to perform inference. In the case of dimensioanl label the output is 3 values instead of a one-hot encode.

> Output of the testing dataloader code:
![image](https://github.com/user-attachments/assets/98835f6a-bafc-482b-b5be-65f76e5cea04)

### Dataset:
> The dataset for categorical training is slightly smaller than the dimensional one because of "non-consensus-labels"

> The test audios are not in label dataframes. In the website we have an example of test files with all the 3200 filenames ids, so I expect that we must keep the order. All files are in our Audio corpus.
