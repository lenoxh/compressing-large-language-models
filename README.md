# Compressing Large Language Models: LoRA Experiments

**Historical project (2024).** 
This repository preserves a completed course project and its original report, example code, and saved outputs. 
The notebooks have not been revalidated in a current environment. 
It is shared as a record of the work, rather than a maintained reproduction package.

This course project addresses the question: 
**How can we make large language models more broadly accessible?** 
The assignment asked us to review LLM compression methods, consider Tim Dettmers' work and ideas from CNN compression, 
and propose an approach of our own.

The [project report](AML_project_report.pdf) reviews pruning, LLM.int8(), and LoRA. 
The assignment's starting references included [Tim Dettmers' work](https://timdettmers.com/)
and [Deep Compression](https://arxiv.org/abs/1510.00149).

Our experimental proposal was to extend attention-layer LoRA to GPT-2's MLP layers and compare the configurations on WebNLG. 
A supplementary MNIST experiment explores LoRA rank in a pure MLP. 
These experiments investigate parameter-efficient adaptation as one aspect of accessibility; 
they do not measure reductions in inference model size or memory usage.

The report contains the full methodology and historical results. 
The notebooks provide example workflows for generating experimental data, 
rather than implementations of every compression method reviewed. 
Known reproduction limitations are documented below.

## Experiments

| Notebook | Description |
|---|---|
| [lora_gpt2_webnlg.ipynb](lora_gpt2_webnlg.ipynb) | Fine-tunes GPT-2 Small and Medium on WebNLG, generates text using beam search, and evaluates BLEU, METEOR, and TER. |
| [lora_mlp_mnist.ipynb](lora_mlp_mnist.ipynb) | Trains a baseline MNIST MLP, applies LoRA to its middle hidden layer, and plots classification accuracy across adapter ranks. |

## Running the notebooks

Start with MNIST for a CPU example. 
For GPT-2, select one configuration and follow it through training, generation, and evaluation. 
The retained GPT-2 cells request five training epochs; there is no implemented quick-run mode.
Shorter runs can demonstrate the pipeline but will not reproduce report scores.

### GPT-2 / WebNLG

The notebook uses Microsoft’s LoRA implementation and includes setup commands to download 
dependencies, pretrained checkpoints, datasets, and evaluation scripts.

1. Open the notebook in a GPU-enabled environment.
2. Replace `/content/drive/MyDrive/...` paths with your own output paths,
   or configure Google Drive in Colab.
3. Run the setup and desired training cells.
4. Update evaluation commands to use the checkpoint produced by your run,
   with matching model size and LoRA rank.
5. Run generation, decoding, and evaluation.

### MNIST / MLP

The notebook uses PyTorch, torchvision, loralib, NumPy, pandas, and Matplotlib.

1. Provide the MNIST CSV files at:
   - `sample_data/mnist_train_small.csv`
   - `sample_data/mnist_test.csv`
2. Run the cells in order to train the baseline and sweep LoRA ranks.
3. Inspect the accuracy plot and trainable-parameter counts.

The notebook writes `model.pth` and `accuracy.png`.

## Project report

See [the project report](AML_project_report.pdf) for the accompanying write-up.

## Known limitations

These historical examples have not been revalidated and may require setup or code changes to run.
Datasets, trained checkpoints, and some original implementation changes are not included.
See the notebooks for experiment-specific notes.
