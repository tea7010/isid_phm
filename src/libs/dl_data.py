import urllib.request
import zipfile
import sys
import os


def download_unzip_data(DEFAULT_DIR='data', re_download=False):
    DL_DATA = {
        'train': {'url': r'https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/Train%20Files.zip',
                  'dirname': os.path.join(DEFAULT_DIR, 'train.zip')},
        'test': {'url': r'https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/Test%20Files.zip',
                 'dirname': os.path.join(DEFAULT_DIR, 'test.zip')},
        'sample_submit': {'url': r'https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/csv.php',
                          'dirname': os.path.join(DEFAULT_DIR, 'sample_sub.csv')},
    }
    _download_data_zip(DEFAULT_DIR, DL_DATA, re_download)
    _unzip(DEFAULT_DIR, DL_DATA)


def _download_data_zip(target_dir, data_dict, re_download=False):
    '''
    データをダウンロードする関数
    とりあえずレポジトリのホームで実行する想定
    '''

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for data in data_dict.keys():
        url = data_dict[data]['url']
        filename = data_dict[data]['dirname']

        # すでにある場合はスキップ
        if os.path.exists(filename):
            if re_download:
                urllib.request.urlretrieve(url, filename)
        else:
            urllib.request.urlretrieve(url, filename)


def _unzip(target_dir, data_dict):
    for data in data_dict.keys():
        filename = data_dict[data]['dirname']

        # zipの解凍
        if filename.split('.')[-1] == 'zip':
            _zip = zipfile.ZipFile(filename)
            _zip.extractall(target_dir)
            _zip.close()
