# OSLMF for CapyMOA

Implementing paper "Online Semi-Supervised Learning with Mix-Typed Streaming Features" with CapyMOA framework.

# File

The overall framework of this project is designed as follows
1. The **dataset** file is used to hold the datasets and labels.

2. The **source** file is all the code for the model.

3. The **Result** is for saving relevant results (e.g. CER, Figure).

# Getting Started
1. Clone this repository

```
git clone https://github.com/rpholvichith/OSLMF_for_CapyMOA.git
```

2. Make sure you meet package requirements by running:

```python
pip install -r requirements.txt
```

3. Running OSLMF model

```python
python OSLMF_Cap.py
```

or run the Jupyter Notebook.

# Credits

This work is inspired by the paper of Wu D., Zhuo S., Wang Y., Chen Z., & He Y. (2023). Online Semi-supervised Learning with Mix-Typed Streaming Features. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(4), 4720-4728. [https://doi.org/10.1609/aaai.v37i4.25596](https://doi.org/10.1609/aaai.v37i4.25596)
