# Spam Classifier

E2E ML demo of a spam classifier — including data processing, model training, evaluation, and serving via Flask.

## Project Organization

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks to train, evaluate and save the classifier model
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
├── setup.py           <- Makes project pip installable (`pip install -e .`) so `src` can be imported.
├── src                <- Source code for use in this project.
│   ├── __init__.py
│   └── app.py         <- Flask API for serving the model.
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

### 2. Set Up a Conda Environment with Python 3.10

```bash
conda create -n spam-env python=3.10
conda activate spam-env
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook (optional)

```bash
jupyter notebook
```

### 5. Run the Flask App (optional)

```bash
cd src
python app.py
```

---

<small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small>