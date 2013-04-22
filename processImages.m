%make raw images to matrix fea and labels gnd.
%columns of fea are normalized straightened images.
%
%have CroppedYale folder from
%http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip and
%getpmgraw.m from http://cvc.yale.edu/projects/yalefacesB/getpgmraw.m in
% the same folder as this file and run. Variables are saved in
% extended_yale_B.mat
%
%to look at images run imagesc(reshape(fea(:,15),h,w)); colormap gray

dbstop error

clearvars
close all

h = 192;
w = 168;
hw = h*w;

%Choose downsampleRatio per dimension (1 = no downsampling)
downsampleRatio = 1;
fea = []; %stores images as normalized vectors in columns 
gnd = []; %stores class number for corresponding column in fea
fileName = []; %stores fileName for the corresponding column in fea
count = 0;

for dd = 1:39
    dd
    if dd~=14 %missing for some reason
        if dd<10
            dirName = ['./CroppedYale/yaleB0' num2str(dd) '/'];
        else
            dirName = ['./CroppedYale/yaleB' num2str(dd) '/'];
        end
            files = dir(dirName);
            for ff = 3:length(files)
               if strcmp(files(ff).name(end-2:end) , 'pgm') && ~strcmp(files(ff).name(end-10:end),'Ambient.pgm')
                   img = getpgmraw([dirName files(ff).name]);
                   if downsampleRatio ~= 1
                       img = downsample(img',downsampleRatio);
                       img = downsample(img',downsampleRatio);
                   end
                   count = count+1;
                   fea = [fea, img(:)/norm(img(:))];
                   gnd = [gnd; dd];
                   fileName{count} = [dirName files(ff).name];
                   
               end      
            end
       
    end       
end

[h w] = size(img);

if downsampleRatio ~=1
save(['extended_yale_B_downsampledBy' num2str(downsampleRatio) '.mat'], 'fea','gnd','h','w','downsampleRatio')
else
    save('extended_yale_B.mat', 'fea','gnd','fileName','h','w','downsampleRatio')
end
