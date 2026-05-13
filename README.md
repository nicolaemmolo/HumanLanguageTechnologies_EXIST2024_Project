# Human Language Technologies – EXIST 2024 - Sexism Categorization in Tweets (Team Medusa)

Human Language Technologies (HLT) project for EXIST 2024 Challenge. Computer Science Master Degree, University of Pisa. A.Y 2023/2024


## Project Description

Sexism remains a pervasive issue that significantly hinders women’s progress in various aspects of life, manifesting particularly severely online in the form of misogyny, abuse, and threats. This project was developed to participate in the **EXIST 2024** (sEXism Identification in Social neTworks) challenge as part of CLEF, with the aim of automatically detecting and classifying sexist content on social media.

Our team ("Medusa") focused on **Task 3: Sexism Categorization in Tweets**. The task required not only the identification of sexist tweets but also their categorization through a hierarchical, multi-class, and multi-label structure.

### Methodology and Approach

To address the complex and subjective nature of the task, we adopted the following strategies:

* **Transformer Architectures:** We trained Transformer-based systems using "Binary Relevance" and "Classifier Chain" models to effectively handle multiple labels.
* **Learning with Disagreements (LeWiDi):** Instead of using a single aggregated label (gold label), the system learns directly from the original annotations provided by 6 different groups of annotators. This approach allows us to capture the diversity of perspectives and mitigate "label bias" arising from socio-demographic differences.
* **Socio-Demographic Analysis:** The dataset includes parameters such as the annotators' gender, age, ethnicity, education level, and country of residence, enabling an in-depth evaluation of subjectivity in sexism identification.
* **Advanced Metrics:** The evaluation was conducted using the **PyEvALL** library, employing the official **ICM** (Information Contrast Measure) metric and its **ICM-soft** extension, which are ideal for hierarchical classification in contexts of annotator disagreement.

## Repository Structure

The repository is organized into the following main folders:

* **`data/`**: This folder contains the original datasets from the EXIST 2024 challenge used for training and testing the models. 
    * *Note: Due to copyright restrictions related to the challenge data, the contents of this folder cannot be published or distributed openly*.
* **`src/`**: Contains all the source code developed for the project. This directory includes scripts related to:
    * Data cleaning and preprocessing phases (tweet text and annotator metadata).
    * Definition and creation of Transformer-based model architectures.
    * Scripts for training and validation.
    * Scripts for inference and the generation of final test files.

## Results and Ranking

The developed models, named **"RoBEXedda"**, demonstrated top-tier performance in the official competition:

* Team **Medusa** achieved **second place** overall in the challenge.
* In the specific "Task 3 Soft-Soft ALL" ranking, our three submissions (runs) secured the **fourth, fifth, and sixth positions**, respectively.

---
*For further technical details and an in-depth analysis of the results, please refer to the official paper: [RoBEXedda at EXIST 2024](https://ceur-ws.org/Vol-3740/paper-88.pdf).*
