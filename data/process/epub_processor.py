import argparse
import re
import os
import shutil
import warnings
import random

import ebooklib
from ebooklib import epub
from tqdm import tqdm
from bs4 import BeautifulSoup

DATA_ROOT_PATH = '.'
RAW_DATA_PATH = os.path.join(DATA_ROOT_PATH, "raw")
READY_DATA_PATH = os.path.join(DATA_ROOT_PATH, "ready")

parser = argparse.ArgumentParser()
parser.add_argument("--author", type=str, default='all')
parser.add_argument("--train_split_ratio", type=float, default=1.0)
parser.add_argument("--max-length", type=int, default=512)


class DocSaver:
    def __init__(self, args, path, mode='para'):
        self.args = args
        self.path = path
        self.mode = mode
        self._split = None
        self._counter = 0

    def set_split(self, split):
        self._split = split
        self._counter = 0

    def save(self, text):
        if self._split is None:
            raise RuntimeError("Attribute split should be set first!")
        path = os.path.join(self.path, self._split, f"{self.mode}-{self._counter}.txt")
        with open(path, 'w') as file:
            file.write(text)
        self._counter += 1

    def save_datasets(self, trainsets, testsets):
        self.set_split('train')
        for sample in trainsets:
            self.save(sample)
        self.set_split('test')
        for sample in testsets:
            self.save(sample)


def train_test_split(sets, ratio, shuffle=True):
    if shuffle:
        random.shuffle(sets)
    cut = int(ratio * len(sets))
    train = sets[:cut]
    test = sets[cut:]
    return train, test


def cut_trailing(text, trailing):
    while text:
        if text[-1] == trailing:
            print("remove trailing:", text)
            text = text[:-1]
        else:
            break
    return text


def create_data_dir_may_recreate(args):
    author_name = args.author
    AUTHOR_PATH = os.path.join(READY_DATA_PATH, author_name)
    if os.path.exists(AUTHOR_PATH):
        shutil.rmtree(AUTHOR_PATH)
    os.mkdir(AUTHOR_PATH)
    os.mkdir(os.path.join(AUTHOR_PATH, "train"))
    if args.train_split_ratio < 1.0:
        os.mkdir(os.path.join(AUTHOR_PATH, "test"))

    return AUTHOR_PATH


def process_moyan(args):
    print("processing mo_yan ...")
    AUTHOR_PATH = create_data_dir_may_recreate(args)

    book = epub.read_epub(os.path.join(RAW_DATA_PATH, "MoYan.epub"))

    def check_string(string: str, pattern: str):
        res = re.match(pattern, string)
        return bool(res)

    def check_para(para):
        text = para.get_text()
        return ("contenttitle" not in para['class']) and text and "电子书" not in text

    doc_saver = DocSaver(args, AUTHOR_PATH)
    collections = []
    for i, doc in tqdm(enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))):
        if not check_string(doc.get_name(), "text/part[0-9]*.html"):
            print(f"doc {doc.get_name()} does not match!")
            continue
        soup = BeautifulSoup(doc.get_body_content().decode('utf-8'), 'html.parser')
        text = []
        for para in soup.find_all('p'):
            if check_para(para):
                txt = para.get_text()
                txt = cut_trailing(txt, '\n')
                if txt:
                    text.append(txt)
        collections.extend(text)

    train_collections, test_collections = train_test_split(collections, args.train_split_ratio)
    doc_saver.save_datasets(train_collections, test_collections)
    # print("collect completed!")

    # with open(os.path.join(READY_DATA_PATH, "moyan.txt"), "w") as file:
    #     file.writelines(collections)


def process_zhangailing(args):
    print("processing zhang_ailing")
    AUTHOR_PATH = create_data_dir_may_recreate(args)

    book = epub.read_epub(os.path.join(RAW_DATA_PATH, "zhangailing.epub"))

    def clean(text):
        if ("书名：" in text) or ("作者：" in text) or ("出版社：" in text) or ("出版日期：" in text) or ("ISBN：" in text) or ("版权所有·侵权必究" in text) or ("有限公司提供授权" in text):
            return ''
        else:
            text = text.replace('\n', '').replace('\r', '')
            return text

    doc_saver = DocSaver(args, AUTHOR_PATH)
    collections = []
    num_trailing_newlines = 0
    for i, doc in tqdm(enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))):
        soup = BeautifulSoup(doc.get_body_content().decode('utf-8'), 'html.parser')
        text = []
        for para in soup.find_all('p'):
            if 'bodytext' in para.get_attribute_list('class'):
                txt = para.get_text()
                txt = cut_trailing(txt, '\n')
                if txt:
                    text.append(txt)
                    # doc_saver.save(txt)
        # text = [t + '\n' for t in text if t]
        collections.extend(text)

    train_collections, test_collections = train_test_split(collections, args.train_split_ratio)

    doc_saver.save_datasets(train_collections, test_collections)

    # print("trailing newlines number", num_trailing_newlines)
    # with open(os.path.join(READY_DATA_PATH, "zhangailing.txt"), 'w') as file:
    #     file.writelines(collections)


process_funcs = {
    "mo_yan": process_moyan,
    "zhang_ailing": process_zhangailing
}


def main():
    args = parser.parse_args()
    if args.author == "all":
        for author, func in process_funcs.items():
            args.author = author
            func(args)
    else:
        process_funcs[args.author](args)
    print("all completed!")


if __name__ == "__main__":
    main()
