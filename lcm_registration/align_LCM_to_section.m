
%align lcm brain images to brain images used for registration

%directory where LCM files and rois are (doesn't matter if path for rois or
%registered sections don't exist yet - you will create it)
section = 'S061' % change this depending on which section you're looking at
file_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/tubes/tubes/'
roi_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/rois'
section_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/sections/'
registered_dir = '/Volumes/lab-znamenskiyp/home/shared/projects/turnerb_V1_MAPseq/BRAC8198.6a/LCM_registration/registered_lcm'

files = dir(fullfile(file_dir, sprintf('%s*', section)))

if ~exist(roi_dir, 'dir')
    mkdir(roi_dir);
    disp(['ROI directory created: ' roi_dir]);
else
    disp(['ROI directory already exists: ' roi_dir]);
end

if ~exist(registered_dir, 'dir')
    mkdir(registered_dir);
    disp(['Registered directory created: ' registered_dir]);
else
    disp(['Registered directory already exists: ' registered_dir]);
end

rois = dir (roi_dir)
for k = 1(files)
    
original = imread(sprintf('%s%s.tif', section_dir, section));
original = rgb2gray(original)

toread = files(k).name;
lcm1 = strcat(file_dir, toread)
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
[filepath,name,ext] = fileparts(toread)
savelcm = strcat(registered_dir, '/', name, '.jpg')
imwrite (recovered, savelcm)

figure;
imshowpair(original,recovered,'montage')

%load image for ROI selection for tube
C = imfuse(original,recovered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);


imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)

tosave = strcat(roi_dir, '/', name, '.png')
imwrite (BW, tosave)
recovered1 = recovered
end
%now do for loop using previous registered LCM image as comparison for ROI
%selection
for k = 2: length(files)
original = imread(sprintf('%s%s.tif', section_dir, section));
original = rgb2gray(original)

toread = files(k).name;
lcm1 = strcat(file_dir, toread)
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
[filepath,name,ext] = fileparts(toread)
savelcm = strcat(registered_dir, '/', name, '.jpg')
imwrite (recovered, savelcm)

figure;
imshowpair(original,recovered,'montage')


%load image for ROI selection for tube
C = imfuse(recovered1,recovered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);


imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)

tosave = strcat(roi_dir, '/', name, '.png')
imwrite (BW, tosave)
recovered1 = recovered
end

close all
clear all

