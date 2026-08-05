# Sarvam-1 Jacobian Lens Experiments

A Kaggle-ready interpretability experiment using **Sarvam-1** and Anthropic’s **Jacobian Lens** to study how an Indian language model handles irrelevant information and instruction suppression in Sanskrit and Hindi.

The experiment examines whether a distractor statement remains visible in the model’s intermediate representations while it generates a response to an unrelated cooking question.

## Experiment

The current notebook uses the following Sanskrit prompt:

```text
पर्वते धूमो नास्ति। एतद् वाक्यम् उपेक्ष्यताम्।
पञ्चमेल-दाल-व्यञ्जनं कथं निर्मीयते?
आवश्यकसामग्रीं तथा क्रमबद्धां पाकविधिं वर्णयतु।
```

Approximate English meaning:

```text
There is no smoke on the hill. Ignore this statement.
How is Panchamel dal prepared?
Describe the required ingredients and the cooking method step by step.
```

A control prompt removes the smoke-related distractor:

```text
पञ्चमेल-दाल-व्यञ्जनं कथं निर्मीयते?
आवश्यकसामग्रीं तथा क्रमबद्धां पाकविधिं वर्णयतु।
```

The comparison helps examine whether the model suppresses the irrelevant statement only in its output or also reduces its influence across intermediate layers.

## Research Questions

This experiment explores four questions:

1. Does Sarvam-1 produce a recipe-focused completion after being instructed to ignore the smoke statement?
2. Do smoke-related concepts remain visible in intermediate model layers?
3. At which layers do cooking-related tokens begin to dominate?
4. How does the internal representation differ between the distractor and control prompts?

## Model

The notebook uses:

```text
sarvamai/sarvam-1
```

Sarvam-1 was selected because it is compact enough for experimentation on Kaggle GPUs and supports Indian-language text using a standard decoder-only transformer architecture.

Sarvam-1 is primarily a text-completion model rather than a fully instruction-tuned chat model. The prompt is therefore formatted as a question-and-answer continuation:

```text
प्रश्नः
...

उत्तरम्:
पञ्चमेल-दाल-व्यञ्जनस्य निर्माणाय
```

The answer stem encourages the model to continue with recipe-related content.

## What Is the Jacobian Lens?

The Jacobian Lens is an interpretability method for examining which vocabulary predictions are encoded within intermediate model representations.

A standard logit lens projects a hidden state directly into vocabulary space. The Jacobian Lens instead estimates how changes in an intermediate representation affect the model’s final output logits.

Conceptually:

```text
Intermediate activation
        ↓
Average downstream Jacobian
        ↓
Estimated final-token influence
        ↓
Vocabulary ranking
```

This makes it possible to inspect which concepts the model appears prepared to express at different positions and layers.

## Notebook Workflow

The notebook performs the following steps:

1. Installs the required libraries.
2. Loads Sarvam-1 on a Kaggle T4 GPU.
3. Creates distractor and control prompts.
4. Generates deterministic baseline completions.
5. Inspects Sanskrit tokenization.
6. Fits a Jacobian Lens using a small generic Indian-language corpus.
7. Compares the Jacobian Lens with the vanilla logit lens.
8. Tracks selected Sanskrit token rankings across layers.
9. Builds an interactive position-by-layer visualization.
10. Saves the fitted lens, generated responses, and visualization.

## Probe Tokens

The experiment tracks tokens associated with the distractor and the requested recipe.

### Distractor-related probes

```text
धूम     smoke
अग्नि    fire
```

### Recipe-related probes

```text
दाल        dal
जल         water
मसाला      spice
चणक        chickpea
सामग्री    ingredients
पाकविधि    cooking method
```

Because Sarvam-1 may split Sanskrit words into multiple tokenizer pieces, the notebook records the exact decoded token piece used for each rank calculation.

## Kaggle Requirements

Recommended environment:

| Requirement | Configuration                     |
| ----------- | --------------------------------- |
| Platform    | Kaggle Notebooks                  |
| Accelerator | GPU T4 or GPU T4 x2               |
| GPU used    | `cuda:0`                          |
| Internet    | Enabled                           |
| Python      | Kaggle default Python environment |
| Model dtype | `float16` on T4                   |

Only one T4 GPU is required. When Kaggle provides two T4 GPUs, the notebook uses the first GPU.

## Installation

The notebook installs the dependencies directly:

```python
%pip install -q --upgrade \
    "transformers>=5.5,<6" \
    accelerate \
    sentencepiece \
    safetensors \
    pandas

%pip install -q \
    "git+https://github.com/anthropics/jacobian-lens.git"
```

## Running the Notebook

1. Create or open a Kaggle notebook.
2. Open **Notebook options**.
3. Select **GPU T4 x2** or an available T4 GPU.
4. Enable Internet access.
5. Upload or import the notebook.
6. Run all cells in order.

The notebook performs a small CUDA operation before downloading the model. This detects incompatible GPU or PyTorch configurations early.

Expected preflight output:

```text
GPU: Tesla T4
Compute capability: 7.5
Model dtype: torch.float16
CUDA kernel preflight: PASSED
```

## Fitting Profiles

The notebook provides two fitting profiles.

### Quick Profile

```python
RUN_PROFILE = "quick"
```

The quick profile uses:

* Two fitting prompts
* Every fourth transformer layer
* Shorter sequences
* Smaller fitting batches

Use this profile to verify that the full workflow runs successfully.

### Research Profile

```python
RUN_PROFILE = "research"
```

The research profile uses:

* The full fitting corpus
* Every second transformer layer
* Longer sequences
* A denser layer-level analysis

Use this profile after validating the notebook with the quick profile.

The research profile is still a compact demonstration. Stronger interpretability conclusions require a larger and more diverse fitting corpus.

## Generated Outputs

The notebook saves its results under:

```text
/kaggle/working/sarvam1_jlens/
```

Generated files include:

```text
*.lens.pt
*.fit_checkpoint.pt
*.interactive.html
*.results.json
```

### Lens Checkpoint

```text
*.lens.pt
```

Contains the fitted average Jacobian Lens.

### Fitting Checkpoint

```text
*.fit_checkpoint.pt
```

Allows an interrupted fitting run to resume.

### Interactive Visualization

```text
*.interactive.html
```

Provides a position-by-layer vocabulary-ranking view.

### Experiment Results

```text
*.results.json
```

Contains:

* Model identifier
* GPU information
* Distractor prompt
* Control prompt
* Generated completions
* Fitted source layers
* Lens location
* Visualization location

## Interactive Visualization

The interactive page shows vocabulary rankings across:

* Prompt positions
* Transformer layers
* Generated continuation positions

Clicking a cell displays the highest-ranked vocabulary tokens for that position and layer.

The visualization pins selected smoke-related and cooking-related token pieces so their movement can be followed even when they do not appear among the top-ranked predictions.

## Interpreting the Results

Three patterns are particularly useful.

### 1. Output Suppression

Check whether the model’s final completion focuses on dal preparation without returning to the hill or smoke statement.

### 2. Intermediate Persistence

A distractor may disappear from the final response while remaining visible in earlier or middle layers.

This would suggest that output-level instruction following does not necessarily imply immediate removal of the irrelevant concept from internal processing.

### 3. Recipe Formation

Track when recipe-related vocabulary begins to rise.

A possible progression could look like:

```text
Prompt comprehension
        ↓
Distractor retained briefly
        ↓
Task selection
        ↓
Recipe concepts activated
        ↓
Cooking instructions generated
```

The actual pattern must be determined from the vocabulary rankings and interactive visualization.

## Jacobian Lens and Logit Lens Comparison

The notebook compares two readout methods.

### Vanilla Logit Lens

```text
Hidden state → output vocabulary projection
```

This assumes an intermediate hidden state can be decoded using the model’s final output projection.

### Jacobian Lens

```text
Hidden state
    → estimated downstream transformation
    → output vocabulary influence
```

The Jacobian Lens attempts to account for how downstream layers transform the representation before final token prediction.

Differences between these rankings may reveal concepts that are causally relevant to later predictions but not directly readable through the standard output projection.

## Important Limitations

This repository is an exploratory interpretability experiment.

It does not establish that:

* The model consciously understands the prompt
* Token rankings represent human-like thoughts
* A high-ranked concept is necessarily used in generation
* An ignored concept has been completely removed
* A small fitted lens generalizes across all Sanskrit prompts
* Intermediate representations correspond directly to natural-language concepts

The Jacobian Lens provides a vocabulary-based view of model representations. It is one measurement method and should be combined with behavioral testing, activation interventions, ablations, and larger evaluation sets.

## Sanskrit Considerations

Sarvam-1 may have stronger coverage for Hindi and other high-resource Indian languages than for classical Sanskrit.

Potential issues include:

* Sanskrit words splitting into several tokenizer pieces
* Hindi forms receiving higher probability than Sanskrit equivalents
* Mixed Hindi and Sanskrit completions
* Transliteration or spelling variation
* English cooking terms appearing in the output

For this reason, the notebook displays token IDs and decoded pieces before interpreting token rankings.

## Suggested Extensions

Possible follow-up experiments include:

### Language Comparison

Run semantically equivalent prompts in:

* Sanskrit
* Hindi
* Malayalam
* Tamil
* Telugu
* English

Compare where distractor-related concepts decline across layers.

### Suppression Strength

Test several instruction forms:

```text
एतद् वाक्यम् उपेक्ष्यताम्।
Ignore this statement.
इस कथन को अनदेखा करें।
```

### Distractor Position

Move the irrelevant statement to:

* The beginning
* The middle
* Immediately before the requested answer
* A previous conversational turn

### Semantic Distractors

Compare unrelated and related distractors:

```text
There is no smoke on the hill.
The dal has a smoky flavour.
The kitchen is filled with smoke.
```

### Intervention Experiments

Promote or suppress selected token directions and observe whether the generated recipe changes.

## Repository Structure

A suggested repository structure is:

```text
.
├── notebooks/
│   └── sarvam1_jacobian_lens_sanskrit_kaggle.ipynb
├── outputs/
│   └── README.md
├── assets/
│   └── screenshots/
├── LICENSE
└── README.md
```

Large model files, fitted lens checkpoints, and generated HTML pages should normally be excluded from Git using `.gitignore`.

Example:

```gitignore
*.pt
*.bin
*.safetensors
*.fit_checkpoint.pt
*.interactive.html
.ipynb_checkpoints/
__pycache__/
```

## Reproducibility

The notebook uses:

```python
set_seed(17)
```

Baseline generation uses greedy decoding:

```python
do_sample=False
```

This makes the generated comparison more reproducible, although results may still differ across:

* PyTorch versions
* Transformers versions
* GPU architectures
* Model revisions
* Jacobian Lens revisions
* Floating-point kernels

For stricter reproducibility, pin the exact package versions and Sarvam-1 model revision.

## Model and Code Licensing

Sarvam-1 is distributed under the license specified on its Hugging Face model page. Review the model license before using the weights or derived artifacts outside research and evaluation.

The Jacobian Lens implementation is governed by the license in its source repository.

This project does not redistribute Sarvam-1 model weights.

## References

1. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)

2. [Verbalizable Representations Form a Global Workspace in Language Models, arXiv](https://arxiv.org/abs/2607.15495)

3. [Anthropic Jacobian Lens Repository](https://github.com/anthropics/jacobian-lens)

4. [Sarvam-1 Model Card](https://huggingface.co/sarvamai/sarvam-1)

5. [Sarvam AI](https://www.sarvam.ai/)

## Disclaimer

This repository is intended for research, education, and experimentation.

The experiment studies model representations and instruction handling. It does not make claims about machine consciousness, subjective experience, or human-equivalent reasoning.
