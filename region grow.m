clc;clear all;close all;
A=imread('15.jpg');
x=rgb2gray(A);
x=wiener2(x);
imshow(x);