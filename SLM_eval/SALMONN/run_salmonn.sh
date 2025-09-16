conda env create -f environment.yml   # creates a new env with the name inside the YAML

python SALMONN/cli_inference_dataset.py --cfg-path SALMONN/configs/decode_config.yaml \
    --input-dir ../EMIS_dataset/ \
    --output-csv {CHOOSE YOU OUTPUT FILE} \
    --prompt "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral."

