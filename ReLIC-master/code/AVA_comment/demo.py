import torch
import clip
from PIL import  Image
import os
import re
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#
#
# def pre_caption(caption, max_words=50):
#     caption = re.sub(
#         r"([.!\"()*#:;~])",
#         ' ',
#         caption.lower(),
#     )
#     caption = re.sub(
#         r"\s{2,}",
#         ' ',
#         caption,
#     )
#     caption = caption.rstrip('\n')
#     caption = caption.strip(' ')
#
#     # truncate caption
#     caption_words = caption.split(' ')
#     if len(caption_words) > max_words:
#         caption = ' '.join(caption_words[:max_words])
#
#     return caption
#
#
# if __name__ == '__main__':
#     device = 'cuda:2' if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
#
#     # image = preprocess(Image.open('/data/yuhao/dataset/AVADataset/1000.jpg')).unsqueeze(0).to(device)
#     f = open('/data/yuhao/dataset/AVA_Comment_Dataset/1000.txt', 'r', encoding='utf-8')
#     comment = ''
#     for c in f.readlines():
#         comment += c.strip('\n')
#     # comment = comment[0]
#
#     text = pre_caption(comment)
#     text = clip.tokenize(text).to(device)
#
#     model.eval()
#     with torch.no_grad():
#         # image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#
#         print(text_features)

# print(len(df))
# f = open('/data/yuhao/Aesthetics_Quality_Assessment/code/AVA_comment/loss_pic.txt', 'r', encoding='utf-8')
# data = f.readline().split()
#
# print(len(data))
# f.close()
# for i in range(len(df)):
#     row = df.iloc[i]
#     image_id = row['image_id']
#     if not os.path.exists(os.path.join(comment_path, f'{image_id}.txt')):
#         f.write(str(i) + ' ')

# lose_id = []
# for i in range(len(img_id)):
#     if not os.path.exists(os.path.join(comment_path, f'{img_id[i]}.txt')) or not os.path.exists(os.path.join(img_path, f'{img_id[i]}.jpg')):
#         lose_id.append(i)
#
# for i in lose_id[::-1]:
#     df = df.drop(axis=0, index=i)
#
# df.to_csv('test.csv')
# comment_path = os.path.join(self.comment_path, f'{image_id}.txt')
# f = open(comment_path, 'r', encoding='utf-8')
# comment = ''
# for c in f.readlines():
# comment += c.strip('\n')
# caption = self.pre_caption(comment)
#
# return image, caption, p.astype('float16')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_score(y_pred):
    w = np.linspace(1, 10, 10)
    score = (y_pred * w).sum()
    return score


def get_score_photo(y_pred):
    w = np.linspace(1, 7, 7)
    score = (y_pred * w).sum()
    return score


if __name__ == '__main__':
    # file_path = '/data/yuhao/Aesthetics_Quality_Assessment/data/AVA/test_.csv'
    # file_path = '/data/yuhao/Aesthetics_Quality_Assessment/data/AVA/test.csv'
    file_path = '/data/yuhao/Aesthetics_Quality_Assessment/data/photonet/test.csv'
    df = pd.read_csv(file_path)
    file_path1 = '/data/yuhao/Aesthetics_Quality_Assessment/data/photonet/train.csv'
    # file_path1 = '/data/yuhao/Aesthetics_Quality_Assessment/data/AVA/train.csv'
    df1 = pd.read_csv(file_path1)

    file_path2 = '/data/yuhao/Aesthetics_Quality_Assessment/data/photonet/val.csv'
    df2 = pd.read_csv(file_path2)

    score_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()
        score_list.append(np.array(get_score_photo(p), dtype='float16'))

        # score_list.append(np.array(score, dtype='float16'))

    for i in range(len(df1)):
        row = df1.iloc[i]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()
        score_list.append(np.array(get_score_photo(p), dtype='float16'))


    for i in range(len(df2)):
        row = df2.iloc[i]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()
        score_list.append(np.array(get_score_photo(p), dtype='float16'))

    print(len(score_list))
    score_list = np.array(score_list)
    plt.hist(score_list, bins=30, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("Average Aesthetics Ratings")
    # 显示纵轴标签
    # plt.ylabel("频数/频率")
    # 显示图标题
    # plt.title("频数/频率分布直方图")
    plt.show()
    # print(score_list)

        # p = y / y.sum()
        # score = get_score(p)
        # df['score'][i] = score
        # if score > 5.0:
        #     good += 1
        # else:
        #     bad += 1
        # if i % 100 == 0:
        #     print('1')
    # print(d)

    # print(good, bad)



