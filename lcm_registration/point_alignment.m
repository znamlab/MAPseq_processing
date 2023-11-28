% use this script if the automatic registration doesn't work well

%directory where LCM files and rois are
section = 'S046' % change this depending on which section you're looking at
tube ='109' % the specific tube that isn't registering
tube_before = num2str(str2num(tube)-1)

file_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/tubes/tubes/'
roi_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/rois'
section_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/sections/'
registered_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/registered_lcm'


original = imread(sprintf('%s%s.tif', section_dir, section));
%original = imread('registered_lcm/s018_tube37.jpg');
unregistered = imread(strcat(file_dir, section, '_tube', tube, '.tif'));
onebefore = imread(strcat(registered_dir, '/', section, '_tube', tube_before, '.jpg'));
[mp,fp] = cpselect(unregistered,original,'Wait',true);
t = fitgeotrans(mp,fp,'projective');
Rfixed = imref2d(size(original));
registered = imwarp(unregistered,t,'OutputView',Rfixed);

C = imfuse(onebefore,registered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);
imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)
imwrite (registered, strcat(registered_dir, '/', section, '_', tube, '.png'))
imwrite (BW, strcat(roi_dir, '/', section, '_', tube, '.jpg'))
clear all
close all
