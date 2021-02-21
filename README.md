# **Wav2Lip**: *Accurately Lip-syncing Videos In The Wild*

This code is part of the paper: _A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild_ published at ACM Multimedia 2020. 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs2)](https://paperswithcode.com/sota/lip-sync-on-lrs2?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs3)](https://paperswithcode.com/sota/lip-sync-on-lrs3?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrw)](https://paperswithcode.com/sota/lip-sync-on-lrw?p=a-lip-sync-expert-is-all-you-need-for-speech)

[[Paper]](http://arxiv.org/abs/2008.10010) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/) | [[Demo Video]](https://youtu.be/0fXaDCZNOJc) | [[Interactive Demo]](https://bhaasha.iiit.ac.in/lipsync) | [[Colab Notebook]](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing) | [[ReSyncED] (coming soon)](#)

 <img src="https://drive.google.com/uc?export=view&id=1Wn0hPmpo4GRbCIJR8Tf20Akzdi1qjjG9"/>

----------
**Highlights**
----------
 - Lip-sync videos to any target speech with high accuracy :100:. Try our [interactive demo](https://bhaasha.iiit.ac.in/lipsync).
 - :sparkles: Works for any identity, voice, and language. Also works for CGI faces and synthetic voices.
 - Complete training code, inference code, and pretrained models are available :boom:
 - Or, quick-start with the Google Colab Notebook: [Link](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing). Checkpoints and samples are available in a Google Drive [folder](https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k?usp=sharing) as well. There is also a [tutorial video](https://www.youtube.com/watch?v=Ic0TBhfuOrA) on this, courtesy of [What Make Art](https://www.youtube.com/channel/UCmGXH-jy0o2CuhqtpxbaQgA). Also, thanks to [Eyal Gruss](https://eyalgruss.com), there is a more accessible [Google Colab notebook](https://j.mp/wav2lip) with more useful features. 
 - :fire: :fire: Several new, reliable evaluation benchmarks and metrics [[`evaluation/` folder of this repo]](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) released. Instructions to calculate the metrics reported in the paper are also present.

--------
**Disclaimer**
--------
All results from this open-source code or our [demo website](https://bhaasha.iiit.ac.in/lipsync) should only be used for research/academic/personal purposes only. As the models are trained on the <a href="http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html">LRS2 dataset</a>, any form of commercial use is strictly prohibhited. Please contact us for all further queries.   

Prerequisites
-------------
- `Python 3.6` 
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`. Alternatively, instructions for using a docker image is provided [here](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668). Have a look at [this comment](https://github.com/Rudrabha/Wav2Lip/issues/131#issuecomment-725478562) and comment on [the gist](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668) if you encounter any issues. 
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.

Getting the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator  | Weights of the expert discriminator | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |

Lip-syncing videos using the pre-trained models (Inference)
-------
You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `results/result_voice.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.

##### Tips for better results:
- Experiment with the `--pads` argument to adjust the detected face bounding box. Often leads to improved results. You might need to increase the bottom padding to include the chin region. E.g. `--pads 0 20 0 0`.
- If you see the mouth position dislocated or some weird artifacts such as two mouths, then it can be because of over-smoothing the face detections. Use the `--nosmooth` argument and give another try. 
- Experiment with the `--resize_factor` argument, to get a lower resolution video. Why? The models are trained on faces which were at a lower resolution. You might get better, visually pleasing results for 720p videos than for 1080p videos (in many cases, the latter works well too). 
- The Wav2Lip model without GAN usually needs more experimenting with the above two to get the most ideal results, and sometimes, can give you a better result as well.

Preparing LRS2 for training
----------
Our models are trained on LRS2. See [here](#training-on-datasets-other-than-lrs2) for a few suggestions regarding training on other datasets.
##### LRS2 dataset folder structure

```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```

Place the LRS2 filelists (train, val, test) `.txt` files in the `filelists/` folder.

##### Preprocess the dataset for fast training

```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```
Additional options like `batch_size` and number of GPUs to use in parallel to use can also be set.

##### Preprocessed LRS2 folder structure
```
preprocessed_root (lrs2_preprocessed)
├── list of folders
|	├── Folders with five-digit numbered video IDs
|	│   ├── *.jpg
|	│   ├── audio.wav
```

Train!
----------
There are two major steps: (i) Train the expert lip-sync discriminator, (ii) Train the Wav2Lip model(s).

##### Training the expert discriminator
You can download [the pre-trained weights](#getting-the-weights) if you want to skip this step. To train it:
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality disriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```

To train with the visual quality discriminator, you should run `hq_wav2lip_train.py` instead. The arguments for both the files are similar. In both the cases, you can resume training as well. Look at `python wav2lip_train.py --help` for more details. You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.

Training on datasets other than LRS2
------------------------------------
Training on other datasets might require modifications to the code. Please read the following before you raise an issue:

- You might not get good results by training/fine-tuning on a few minutes of a single speaker. This is a separate research problem, to which we do not have a solution yet. Thus, we would most likely not be able to resolve your issue. 
- You must train the expert discriminator for your own dataset before training Wav2Lip.
- If it is your own dataset downloaded from the web, in most cases, needs to be sync-corrected.
- Be mindful of the FPS of the videos of your dataset. Changes to FPS would need significant code changes. 
- The expert discriminator's eval loss should go down to ~0.25 and the Wav2Lip eval sync loss should go down to ~0.2 to get good results. 

When raising an issue on this topic, please let us know that you are aware of all these points.

Evaluation
----------
Will be updated.

License and Citation
----------
The software can only be used for personal/research/non-commercial purposes. Please cite the following paper if you use this code:
```
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```


Acknowledgements
----------
Parts of the code structure is inspired by this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.


Further Development by Shuyi
----------
While lip-syncing video, you can also compute the cosine similarity between the video and audio embeddings. This could be used to detect and match the voice to the person in any video.
```bash
python inference.py --checkpoint_path <GAN ckpt> --syncnet_checkpoint_path <SyncNet ckpt>  --face <video.mp4> --audio <an-audio-source> 
```

Here are some datapoints:
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --syncnet_checkpoint_path checkpoints/lipsync_expert.pth --face ./sample_data/chinese_shuyi.mp4 --audio ./sample_data/chinese_shuyi.mp4
DescribeResult(nobs=207, minmax=(0.0, 0.9759844), mean=0.52837175, variance=0.06384362, skewness=-0.2030022293329239, kurtosis=-0.9478885536467487)
```
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --syncnet_checkpoint_path checkpoints/lipsync_expert.pth --face ./sample_data/chinese_shuyi.mp4 --audio ./sample_data/japanese_shuyi.mp4
DescribeResult(nobs=212, minmax=(0.0, 0.88642454), mean=0.22506772, variance=0.05906559, skewness=0.853131890296936, kurtosis=-0.46799933113297065)
```
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --syncnet_checkpoint_path checkpoints/lipsync_expert.pth --face ./sample_data/silence_shuyi.mp4 --audio ./sample_data/chinese_shuyi.mp4
DescribeResult(nobs=207, minmax=(0.0, 0.8705838), mean=0.25465956, variance=0.05431626, skewness=0.9279311895370483, kurtosis=0.11473465641982417)
```
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --syncnet_checkpoint_path checkpoints/lipsync_expert.pth --face ./sample_data/japanese_shuyi.mp4 --audio ./sample_data/japanese_shuyi.mp4
DescribeResult(nobs=212, minmax=(0.0, 0.9585642), mean=0.5192854, variance=0.07212567, skewness=-0.32968321442604065, kurtosis=-0.9731913222597024)
```
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --syncnet_checkpoint_path checkpoints/lipsync_expert.pth --face ./sample_data/input_vid.mp4 --audio ./sample_data/japanese_shuyi.mp4
DescribeResult(nobs=179, minmax=(0.0, 0.90442127), mean=0.14771806, variance=0.035665836, skewness=1.5768992900848389, kurtosis=2.204756945157232)
```
