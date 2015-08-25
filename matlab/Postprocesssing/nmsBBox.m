function detection = nmsBBox(bbox, prob, T_nms, flag_nms_percls)

% ----------------------------------------------------------------
% Function for Non-maximum Suppression of Original Bounding Boxes
%
% Input: 1. Original Bounding Boxes;
%        2. Original Probabilities
%        3. Threshold of NMS
%        4. 
% Output: Detection Result containing
%
% Written by Ziwei Liu, 2015/08/12
% ----------------------------------------------------------------