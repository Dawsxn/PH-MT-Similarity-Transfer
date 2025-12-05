# PH-MT-Similarity-Transfer

- First run the scraper notebook
- Then run the preprocessing notebook for data cleaning
- Run the parallel-corpora to generate paired verses
- Run the data-splits notebook to divide the corpora into training, developing, and testing sets.

- For the actual training, just run the train-models-nllb-lora-bf16 notebook
- Make sure to install the needed libraries first

- For evaluation, there are two different evaluation sets
- Run both notebooks to get the evaluation 