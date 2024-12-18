# Local Code FAQ

Just some general questions about the code, that can be (or not) important.

### How dataloader from benchmark works?
>> It works by loading all waveforms in memory, and then passing to 3 datasets: wavs, labels and utterance. The output of dataloader in each iteration is (respectively): waveform, label, waveform_mask, utterance_id. Utterance_Id is basically a string which can be retrieved to perform inference. In the case of dimensioanl label the output is 3 values instead of a one-hot encode.
