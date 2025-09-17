# Evaluating Emotion Recognition of Spoken Language Models on Emotionally Incongruent Speech

> Pedro Corrêa, João Lima, Victor Moreno, Lucas Ueda, Paula Costa

In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results reveal that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.

![model](assets/images/general_pipeline.png)

## Usage

### Generating Speech Samples with each TTS system

Download the ESD dataset (https://github.com/HLTSingapore/Emotional-Speech-Data). We filtered this dataset for English speakers only. You can allocate the filtered version in ```/speech_generation/ESD_english/```.

#### CosyVoice2

```bash

cd speech_generation/cosyvoice

python generante_samples_cosyvoice.py

```

#### F5-TTS

```bash

cd speech_generation/f5tts

TBD

```

#### StyleTTS2

```bash

cd speech_generation/styletts2

python batch_inference.py \
    --esd <path to original ESD dataset> \
    --txts <path to generated emotional sentences> \
    --esd_ser_test <path to SER test_set> \
    --out <output directory>

```

### Evaluating the Spoken Language Models on th Emotion Recognition Task

Download the EMIS dataset (https://ieee-dataport.org/documents/emotionally-incongruent-synthetic-speech-dataset-emis), or use other speech samples to evaluate each SLM. The default path for sample analysis is ```/SLM_eval/EMIS_dataset/```.

Each folder inside ```/SLM_eval/``` directly opens up a fork for all evaluated SLMs repositories, all containing individual ```README.md```, ```environment.yml```, ```requirements.txt``` and added/modified scripts from each original repository.

## Citation

Consider citing this paper if you find our work useful:

```
TBD
```