# Deep learning projects 1 and 5

Module 4, Part 3. Both projects use data that ships with scikit-learn (no downloads) and run on a laptop CPU.

## Setup (conda)
    conda env create -f environment.yml
    conda activate dl-projects
(or: `pip install -r requirements.txt`)

## Project 1 · Courier company · Little data, big model
    cd p1_curves
    python p1_train.py        # about a minute -> curves.png, results.csv
Trains three models on only 200 digit images (small, big, big + dropout + weight decay) and plots training and validation loss.

| Model | Final training loss | Best validation loss | Test accuracy |
|---|---|---|---|
| Small | 1.767 | 1.800 at epoch 300 | 0.751 |
| Big | 0.001 | 0.248 at epoch 84 | 0.934 |
| Big + dropout + decay | 0.006 | 0.234 at epoch 157 (stopped at 187) | 0.937 |

**Ships:** the regularised model: lowest validation loss and it stopped itself 113 epochs early. **Next:** more labelled data, since 200 images is the real limit (the regularised model is only 0.3 points better).

## Project 5 · City Hospital · Live training dashboard
    cd p5_dashboard
    streamlit run app.py
`trainer.py` is the engine (it calls you back after every epoch); `app.py` is the page (sidebar settings, live chart, diagnosis, experiments table with CSV download, check a patient, one-time final test). Data: 569 breast cancer scans, 30 measurements, malignant = 1, split 60 / 20 / 20.
Read the findings in `p5_dashboard/REPORT.md`.

## Limitations
- Project 1 uses 8x8 scanned digits only; photos of handwriting will fail.
- Project 5 is one public dataset: it is a teaching model, not a medical device. A doctor must review every result.
