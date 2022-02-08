# %%
import os
import glob

import seaborn as sns

from paths import *


def count_doc_length(author):
    lengths = []
    path = os.path.join(f"../ready/{author}/**/*.txt")
    print(path)
    files = glob.glob(path)
    for file in files:
        with open(file, 'r') as file:
            content = file.read()
            lengths.append(len(content))
    sns.histplot(lengths)


# %%
count_doc_length("mo_yan")

# %%
