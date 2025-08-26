StyleTTS2 implementation cloned from [the official repository.](https://github.com/yl4579/StyleTTS2)

The original README is available at ```./StyleTTS2/README.md```

We generated expressive speech samples with ESD reference voices using the script at ```./StyleTTS2/batch_inference.py```

Inference script usage:

```bash
python batch_inference.py \
    --esd <path to original ESD dataset> \
    --txts <path to generated emotional sentences> \
    --esd_ser_test <path to SER test_set> \
    --out <output directory>
```

Default is 24000 kHz sample rate and 7 reference samples.