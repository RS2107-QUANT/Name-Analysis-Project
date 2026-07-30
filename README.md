# Trigram Statistical Analysis and Name Generation

This project performs character-level statistical analysis on the `names.txt` dataset from Andrej Karpathy’s **makemore** project. It uses a trigram probability model to learn patterns in English first names and generate new, realistic-sounding names.

## Project Overview

The model analyzes character sequences in approximately **32,000 English first names**. Each name is processed at the character level, and a trigram model learns the probability of the next character based on the previous two characters.

For example, the probability of the next character is represented as:

[
P(c \mid a,b)
]

where:

* `a` and `b` are the previous two characters
* `c` is the next character

## Features

* Character-level analysis of English first names
* Trigram frequency analysis
* 27-character vocabulary consisting of:

  * 26 lowercase English letters
  * 1 boundary token (`.`)
* Trigram count tensor with shape:

```text
27 × 27 × 27
```

* Laplace (add-one) smoothing to prevent zero probabilities
* Temperature-based sampling for controlled randomness
* Generation of novel names not present in the original dataset
* Trigram probability heatmap for visual analysis

## Methodology

### 1. Data Preprocessing

The names are:

* Converted to lowercase
* Stripped of unnecessary whitespace
* Processed using the boundary token `.`

For example:

```text
emma
↓
. . e m m a .
```

The boundary tokens allow the model to learn how names begin and end.

### 2. Trigram Count Model

A three-dimensional count tensor is created:

```text
N[i, j, k]
```

where `N[i, j, k]` represents the number of times character `k` follows the two-character context `(i, j)`.

### 3. Laplace Smoothing

Add-one smoothing is applied to prevent unseen character combinations from receiving a probability of zero:

[
P(c \mid a,b)
=============

\frac{N(a,b,c)+1}
{\sum_c N(a,b,c)+27}
]

### 4. Temperature-Based Sampling

The model uses a sampling temperature of:

```text
0.7
```

A lower temperature produces more conservative and predictable names, while a higher temperature increases randomness and diversity.

### 5. Name Generation

The model generates names one character at a time. Each new character is sampled using the probability distribution associated with the previous two characters.

Generated names are checked against the original dataset to ensure that they are novel.

## Generated Names

Using:

```text
Random Seed: 9663
Temperature: 0.7
```

the model generated:

```text
Aliya
Brennen
Chastine
Davelyn
Emryn
Faylin
Gresley
Harlyn
Isadora
Jaysen
```

## Key Observations

* Common starting characters include `a`, `j`, `k`, `m`, `l`, and `s`.
* Common ending characters include `a`, `n`, `e`, and `y`.
* The model learns common name patterns and character transitions.
* Vowels and consonants frequently alternate, producing more pronounceable names.
* Rare character combinations receive small non-zero probabilities because of Laplace smoothing.
* Temperature scaling helps balance realistic name generation and diversity.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

## Dataset

The dataset is the publicly available `names.txt` file from Andrej Karpathy’s **makemore** repository.

The dataset contains approximately **32,033 first names**, with one name per line.

## Project Structure

```text
karpathy-name-analysis/
│
├── names.txt
├── name_analysis.ipynb
├── trigram_heatmap.png
└── README.md
```

## Limitations

* The trigram model considers only the previous two characters.
* Longer naming patterns and syllable structures are not fully captured.
* Laplace smoothing may assign small probabilities to unrealistic character combinations.
* The model has no understanding of name meaning, gender, or cultural context.
* Generated-name quality is evaluated qualitatively rather than using a formal metric.

## Future Improvements

* Extend the model to 4-gram or 5-gram character models.
* Compare the trigram model with neural language models.
* Implement an RNN or Transformer-based name generator.
* Add perplexity-based evaluation.
* Create an interactive web application with adjustable temperature settings.
* Extend the dataset to include names from multiple languages and cultures.

## References

* Andrej Karpathy — **makemore**
* Andrej Karpathy — **Neural Networks: Zero to Hero**
