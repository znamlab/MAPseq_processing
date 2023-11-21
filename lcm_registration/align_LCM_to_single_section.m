% use this script if you missed a tube roi and want to re-do

%directory where LCM files and rois are
section = 'S059' % change this depending on which section you're looking at
tube ='10' % the specific tube that isn't registering
tube_before = '9'%num2str(str2num(tube)-1) %put as 'NA' if it's the first one

file_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/tubes/tubes/'
roi_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/rois'
section_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/sections/'
registered_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/registered_lcm'

original = imread(sprintf('%s%s.tif', section_dir, section));
if strcmp(tube_before, 'NA')
    onebefore = original
else
    onebefore = imread(strcat(registered_dir, '/', section, '_tube', tube_before, '.jpg'));
end

original = rgb2gray(original)
lcm1= strcat(file_dir, section, '_tube', tube, '.tif');
lcm = imread(lcm1);
lcm = im2gray(lcm)

ptsOriginal  = detectSURFFeatures(original);
ptsLcm = detectSURFFeatures(lcm);

[featuresOriginal,  validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
[featuresLcm, validPtsLcm] = extractFeatures(lcm, ptsLcm);

indexPairs = matchFeatures(featuresOriginal, featuresLcm);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedLcm = validPtsLcm(indexPairs(:,2));

figure;
showMatchedFeatures(original,lcm,matchedOriginal,matchedLcm);

[tform, inlierIdx] = estimateGeometricTransform2D(...
    matchedLcm, matchedOriginal, 'similarity');
inlierLcm = matchedLcm(inlierIdx, :);
inlierOriginal  = matchedOriginal(inlierIdx, :);

figure;
showMatchedFeatures(original,lcm,inlierOriginal,inlierLcm);
title('Matching points (inliers only)');
legend('ptsOriginal','ptsLcm');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc)
thetaRecovered = atan2(ss,sc)*180/pi


outputView = imref2d(size(original));
recovered  = imwarp(lcm,tform,'OutputView',outputView);

savelcm = strcat(registered_dir, '/', section, '_TUBE', num2str(tube), '.jpg')
imwrite (recovered, savelcm)

figure;
imshowpair(original,recovered,'montage')

C = imfuse(onebefore,recovered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);
imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)
imwrite (BW, strcat(roi_dir, '/', section, '_TUBE', num2str(tube), '.png'))
clear all
close all