# Evaluating Spoken Language Models for Speech Emotion Recognition

> Pedro Corrêa, João Lima, Victor Moreno, Lucas Ueda, Paula Costa

In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results reveal that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.

![model](assets/images/general_pipeline.png)

## Usage

### Load Model


### Have fun!

```python
messages = [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": "<path_to_audio_file>"},
            {"role": "user", "content": "Describe the audio."}
        ]

generated_ids = model.chat(
    messages, 
    max_new_tokens=128, 
    do_sample=True, 
    temperature=0.6, 
    top_p=0.9
)

response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```


### Examples


## Citation

if you find our work useful, please consider citing the paper:

```
TBD
```