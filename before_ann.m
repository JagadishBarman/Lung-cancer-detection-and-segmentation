function [block,im_block ] = before_ann(num_block,c,image,lab_3d,im_block,img_bw3d )


   if num_block~=0
                for n=1:50
                    if c(3)==n
                        break;
                    end
                    im_block3=lab_3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)-n)==num_block;
                    if any(any(im_block3(round(c(5)/2)-1:round(c(5)/2),round(c(4)/2)-1:round(c(4)/2))))==1
                         im_block=cat(3,image(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)-n),im_block);
                    else
                        break;
                    end
                end
                for n=1:50
                    if c(3)==n
                        break;
                    end
                    im_block3=lab_3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)+c(6)-1+n)==num_block;
                    if any(any(im_block3(round(c(5)/2)-1:round(c(5)/2),round(c(4)/2)-1:round(c(4)/2))))==1
                        im_block=cat(3,im_block,image(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)+c(6)-1+n));
                    else
                        break;
                    end
                end
            end
            block=img_bw3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3):c(3)+c(6)-1);
%                                     figure;imshow(afterbw(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)))
            im_block2=lab_3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)).*img_bw3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3));
            num_block=im_block2(round(c(5)/2),round(c(4)/2));
            if num_block~=0
                for n=1:50
                    if c(3)==n
                        break;
                    end
                    im_block3=lab_3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)-n)==num_block;
                    if any(any(im_block3(round(c(5)/2)-1:round(c(5)/2),round(c(4)/2)-1:round(c(4)/2))))==1
                        block=cat(3,im_block3,block);
                    else
                        break;
                    end
                end
                for n=1:50
                    if c(3)==n
                        break;
                    end
                    im_block3=lab_3d(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3)+c(6)-1+n)==num_block;
                    if any(any(im_block3(round(c(5)/2)-1:round(c(5)/2),round(c(4)/2)-1:round(c(4)/2))))==1
                        block=cat(3,block,im_block3);
                    else
                        break;
                    end
                end
            end

end

