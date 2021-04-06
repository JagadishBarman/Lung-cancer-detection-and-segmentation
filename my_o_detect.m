function [  p_seed,  y_seed1,  x_seed1 ] = my_o_detect( sp_n)
% sp_n=['solid\9'];
if sp_n(end-2)=='\'
    
    ss=sp_n(end-1:end);
else
    ss=sp_n(end);
end

filename = 'special_datan.xlsx';
subsetA = xlsread(filename,1); 

[x y]=find(subsetA(:,1)==str2num(ss));
p_seed=subsetA(x,2);
y_seed1=round(subsetA(x,3));
x_seed1=round(subsetA(x,4));
end

