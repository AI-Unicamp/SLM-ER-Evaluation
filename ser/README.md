# How to train

Please refer to original emotion2vec repository for instruction on how to fine tune the base model https://github.com/ddlBoJack/emotion2vec.

In order to train on ESD dataset we share our .sh script to run on slurm cluster, which can be adapted to run under other resources. Environment used to run is given by the requirements.txt.

```bash
sh bin/train_esd.sh
```

# Getting results

To make inference with the fine-tuned emotion2vec please refer to the /bin/inference.py script which also have the slurm scripts examples to run.

To extract metrics run:

python bin/consolidate_results.py /path_to/destav2/whatever.csv ./bin/inference/

It will create a folder based on the name of the csv file and store all metrics there.