clear all
close all
original = imread('sections/brain1_s018.jpg');
%original = imread('registered_lcm/s018_tube37.jpg');
unregistered = imread('tubes/s018_tube39.jpg');
onebefore = imread('registered_lcm/s018_tube38.jpg');
[mp,fp] = cpselect(unregistered,original,'Wait',true);
t = fitgeotrans(mp,fp,'projective');
Rfixed = imref2d(size(original));
registered = imwarp(unregistered,t,'OutputView',Rfixed);

C = imfuse(onebefore,registered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);
imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)
imwrite (BW, 'rois/s018_tube39.png')
imwrite (registered, 'registered_lcm/s018_tube39.jpg')
