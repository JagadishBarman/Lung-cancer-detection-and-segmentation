clc;clear all;close all;
A=imread('14.jpg');
x=rgb2gray(A);
y=wiener2(x);
imshow(x)
figure;imshow(y)