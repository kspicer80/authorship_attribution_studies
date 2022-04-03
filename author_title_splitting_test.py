import os
from strip_headers_and_footers import strip_headers


def load_directory(directory):
    documents, authors, titles = [], [], []
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)

        with open(filename.path, encoding='utf-8') as f:
            contents = f.read()
        contents = strip_headers(contents)
        lemmas = contents.lower().split()
        documents.append(' '.join(lemmas))
        authors.append(author[0])
        title = filename.name.replace('.txt', '')
        split_title = title.split('_')[1:]
        rejoined_title = '_'.join(split_title)
        titles.append(f"{rejoined_title}")

    return documents, authors, titles

data_folder = 'cjlw_texts/'

documents, authors, titles = load_directory(data_folder)
print(authors)
print(titles)
