function [ diameters ] = all_diameter( position )
% �p��G�ȼv���W�Ҧ�����
%  position ���G�ȼv���W���I���Ҧ��y�Яx�}
% diameters ���s��Ҧ����פ��x�}


l=length(position);

for tt=1:1:l
for nn=1:1:l
 
diameters(tt,nn)=(sqrt(((position(nn,1)-position(tt,1)).^2)+((position(nn,2)-position(tt,2)).^2)));

end
end


end

