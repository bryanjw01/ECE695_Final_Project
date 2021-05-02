import os
import zipfile 
import gdown



def retrieve_CELEBA():
    data_root = 'data/celeba'

    dataset_folder = f'{data_root}/img_align_celeba'

    url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'

    download_path = f'{data_root}/img_align_celeba.zip'


    if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)


    gdown.download(url, download_path, quiet=False)

    
    with zipfile.ZipFile(download_path, 'r') as ziphandler:
    ziphandler.extractall(dataset_folder)
