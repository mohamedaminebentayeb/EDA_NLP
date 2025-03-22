# LLM-Augmentation: Data Augmentation Using Language Models for Boosting Performance on Text Classification Tasks
[![Conference](http://img.shields.io/badge/EMNLP-2025-4b44ce.svg)](https://arxiv.org/abs/1901.11196)

For a survey of data augmentation in NLP, see this [repository](https://github.com/styfeng/DataAug4NLP/blob/main/README.md) or this [paper](http://arxiv.org/abs/2105.03075).

This repository provides code for augmenting text classification datasets using our novel **LLM-Augmentation** method. Unlike traditional augmentation techniques that rely on simple text editing operations, our approach leverages a pretrained Language Model (LLM) to perform intelligent modifications while preserving the original meaning.

A blog post that explains our LLM-based augmentation method is available [here](https://medium.com/). 

By adapting ideas from EDA and enhancing them with the power of modern LLMs, our method achieves robust performance gains, especially on small datasets (`N < 500`). The LLM handles tasks such as synonym replacement, word insertion, swapping, and deletion by generating modifications according to carefully designed prompts.

Given a sentence in your training set, our method performs the following operations via LLM calls:

- **Synonym Replacement (SR):** The LLM replaces exactly *n* words in the sentence with contextually appropriate synonyms.
- **Random Insertion (RI):** The LLM inserts exactly *n* new words into the sentence at strategic positions to maintain coherence.
- **Random Swap (RS):** The LLM swaps exactly *n* words in the sentence, ensuring that the overall meaning is preserved.
- **Random Deletion (RD):** The LLM selectively deletes words from the sentence based on a specified probability *p*, while keeping the core meaning intact.



# Usage

You can apply our LLM-Augmentation method on any text classification dataset in just a few minutes. Follow these steps:
## Run the Augmentation Script
Our implementation takes the input file and generates augmented sentences using LLM-based operations. Run the script as follows:
```
bash

python code/augment.py --input=your_dataset.txt --output=augmented_dataset.txt --num_aug=16 --alpha_sr=0.05 --alpha_rd=0.1 --alpha_ri=0.1 --alpha_rs=0.1
--num_aug: Number of augmented sentences generated per original sentence.

--alpha_sr, --alpha_ri, --alpha_rs, --alpha_rd: Parameters controlling the extent of augmentation (e.g., percent of words affected).
```
# Citation
If you use EDA in your paper, please cite us:
```
@inproceedings{wei-zou-2019-eda,
    title = "{EDA}: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks",
    author = "Wei, Jason  and
      Zou, Kai",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1670",
    pages = "6383--6389",
}
```



