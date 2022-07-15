from VGG16_test_leo import run_exp as VGG
from Inceptionv3_test_leo import run_exp as InceptionV3
from Inceptionv4_test_leo import run_exp as InceptionV4
from ResNet50_test_leo import run_exp as ResNet50
from DenseNet_test_leo import run_exp as DenseNet


for use_predict in [False]:
    post = 'predict' if use_predict else 'no_predict'
    VGG([[f'./log/VGG x1 {post}/', 5, 1, 16, use_predict], [f'./log/VGG x2 {post}/', 5, 2, 16, use_predict], [f'./log/VGG x3 {post}/', 5, 3, 16, use_predict], [f'./log/VGG x4 {post}/', 5, 4, 16, use_predict]])
    InceptionV3([[f'./log/Inception V3 x1 {post}/', 5, 1, 16, use_predict],[f'./log/Inception V3 x2 {post}/', 5, 2, 16, use_predict], [f'./log/Inception V3 x3 {post}/', 5, 3, 16, use_predict], [f'./log/Inception V3 x4 {post}/', 5, 4, 16, use_predict]])
    InceptionV4([[f'./log/Inception V4 x1 {post}/', 5, 1, 16, use_predict], [f'./log/Inception V4 x2 {post}/', 5, 2, 16, use_predict], [f'./log/Inception V4 x3 {post}/', 5, 3, 16, use_predict], [f'./log/Inception V4 x4 {post}/', 5, 4, 16, use_predict]])
    ResNet50([[f'./log/ResNet x1 {post}/', 5, 1, 16, use_predict], [f'./log/ResNet x2 {post}/', 5, 2, 16, use_predict], [f'./log/ResNet x3 {post}/', 5, 3, 16, use_predict], [f'./log/ResNet x4 {post}/', 5, 4, 16, use_predict]])
    DenseNet([[f'./log/DenseNet x1 {post}/', 5, 1, 16, use_predict], [f'./log/DenseNet x2 {post}/', 5, 2, 16, use_predict], [f'./log/DenseNet x3 {post}/', 5, 3, 16, use_predict], [f'./log/DenseNet x4 {post}/', 5, 4, 16, use_predict]])
    

# batch size experiments
# VGG([[f'./log/VGG bs4/', 5, 1, 4], [f'./log/VGG bs8/', 5, 1, 8], [f'./log/VGG bs32/', 5, 1, 32], [f'./log/VGG bs64/', 5, 1, 64]])
# InceptionV3([[f'./log/Inception V3 bs4/', 5, 1, 4], [f'./log/Inception V3 bs8/', 5, 1, 8], [f'./log/Inception V3 bs32/', 5, 1, 32], [f'./log/Inception V3 bs64/', 5, 1, 64]])
# InceptionV4([[f'./log/Inception V4 bs4/', 5, 1, 4], [f'./log/Inception V4 bs8/', 5, 1, 8], [f'./log/Inception V4 bs32/', 5, 1, 32], [f'./log/Inception V4 bs64/', 5, 1, 64]])
# ResNet50([[f'./log/ResNet bs4/', 5, 1, 4], [f'./log/ResNet bs8/', 5, 1, 8], [f'./log/ResNet bs32/', 5, 1, 32], [f'./log/ResNet bs64/', 5, 1, 64]])
# DenseNet([[f'./log/DenseNet bs4/', 5, 1, 4], [f'./log/DenseNet bs8/', 5, 1, 8], [f'./log/DenseNet bs32/', 5, 1, 32], [f'./log/DenseNet bs64/', 5, 1, 64]])
#
#


