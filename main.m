
function [] = main(path_side_info, path_groundtruth, ...
        scaling_factor, in_type, filename_results,  ...
        varargin)


% Path to utility functions
addpath('utils/')

% ======================================================
% Check inputs

% ------------------------------------------------------
% Check that 'path_side_info' and 'path_groundtruth' have the same
% number of images
ext   =  {'*/.jpg','/*.png','/*.bmp'}; % possible image extensions

files_path_side_info = [...
    dir([path_side_info, ext{1}]), ...
    dir([path_side_info, ext{2}]),  ...
    dir([path_side_info, ext{3}])
    ];

files_path_groundtruth = [...
    dir([path_groundtruth, ext{1}]), ...
    dir([path_groundtruth, ext{2}]),  ...
    dir([path_groundtruth, ext{3}]) 
    ];

if length(files_path_side_info) ~= length(files_path_groundtruth)
    error('Specified folders should have the same number of images')
end

if scaling_factor <= 1 || scaling_factor >= 9
    error('scaling_factor has to be between 1 and 8')
end

% ***********
% Defaults
% ***********
GPU = 0;
SHOW_IMAGES = 0;
SHOW_RESULTS = 0;
A_h  = @A_bicubic;
AT_h = @AT_bicubic;
beta = 1;

% Read optional input
if (rem(length(varargin),2) == 1)
    error('Optional parameters should always go in pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})            
            case 'GPU'
                GPU = varargin{i+1};
            case 'SHOW_IMAGES'
                SHOW_IMAGES = varargin{i+1};
            case 'SHOW_RESULTS'
                SHOW_RESULTS = varargin{i+1};
             case 'BETA'
                beta  = varargin{i+1};
            case 'A_FORWARD'
                A_h = varargin{i+1};
            case 'A_TRANSPOSE'
                AT_h = varargin{i+1};         
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end



% ======================================================

% ======================================================
% Main loop over all images

try
    % Images to be used as side information (w)
    DNNs = ReadImages(path_side_info);

    % Ground truth images (x)
    HR = ReadImages(path_groundtruth);
catch
    error('Something wrong with the folders used as input to main.m')
end

num_images = numel(HR);   % Number of images in the folders

% check if the images have the same size in the correspondig folders
for n = 1:1:num_images
   check(n) =  isequal(size(DNNs(n).data),size(HR(n).data)); 
end

if all(check) == 0
    fprintf('The images in the different folders do not correspond to eachother.')
end

psnr_tvtv = zeros(1, num_images);
ssim_tvtv = zeros(1, num_images);
psnr_cnn  = zeros(1, num_images);
ssim_cnn  = zeros(1, num_images);

 disp(['-----Super-Resolution-----', 'x', num2str(scaling_factor),'-----']);

x_hat = []
parfor j = 1:1:num_images    
    % Image w
    im_gt = struct()
    im_w = struct()
    im_HR = struct()
    im_w.out = DNNs(j).data;
   
    % if the original network is trained on an RGB image, no pre-processing
    % is done
   

    if in_type == 'RGB' 
        w = im_w.out;         
        x = HR(j).data; 
      if size(im_w.out,3) > 1
        im_gt.out = im2double(HR(j).data(:,:,1));
        x =  im2double(reshape(x,[],3));
        w =  im2double(reshape(w, [], 3));
      end
      if size(im_w.out,3) == 1
         w = im_w.out;
         x = HR(j).data;
         im_gt.out = (HR(j).data);
         x =  im2double(x(:));
         w =  im2double(w(:));
      end
    end
         
     if in_type == 'Y' 
      if  size(im_w.out,3) > 1
        im1 = rgb2ycbcr(im_w.out);
        im_w.out = im1(:, :, 1);
        im_HR.out = HR(j).data;
        im1 = rgb2ycbcr(im_HR.out);
        im_gt.out = im1(:, :, 1);
        
       end

    if size(HR(j).data,3) == 1 
        im_gt.out = HR(j).data;
    end
        w = im2double(im_w.out(:)); % normalizing the intensity values
        x = im2double(im_gt.out(:)); % normalizing the intensity values
    end

    % Dimensions of image
    [M, N, channel] = size(im_w.out); 
    if in_type == 'RGB' & size(im_w.out,3) > 1
        channel = 3
    else
       channel = 1
    end
        
    n = M*N;

    % -------------------------------------------------------------

    % ----------------------------------------------
    %% Post-processing step using TVTV Solver
     X_ADMM = zeros(n,channel);
    tic
    for i = 1:channel
         % Obtain the LR image b by sampling x
        b = A_h(x(:,i),scaling_factor,M,N); 
        if GPU 
            [x_ADMM, k_ADMM] = TVTV_Solver_GPU(M, N, b, w(:,i), beta, A_h, AT_h, scaling_factor);
            X_ADMM(:,i) = x_ADMM;
        else
            [x_ADMM, k_ADMM] = TVTV_Solver_CPU(M, N, b, w(:,i), beta, A_h, AT_h, scaling_factor);
            X_ADMM(:,i) = x_ADMM;
        end
        
    end
    toc
    fprintf('Image %i processed \n',j)
    % ----------------------------------------------------------------------------------------------------
   
    
    % ----------------------------------------------------------------------------------------------------
    %% Reshape from vector to matrix and rescale entries to [1,255]
    x_hat = (reshape(X_ADMM,M,N, channel)); 
    x_hat = uint8(x_hat*255); 
    figure
    imshow(x_hat)
    % ----------------------------------------------------------------------------------------------------

    if SHOW_IMAGES==1
        figure
        plotimages(j,im_HR.out, im_gt.out,im_w.out, x_hat, scaling_factor)
    end
    
    if in_type == 'RGB' & size(im_w.out,3) > 1
        x_hat = rgb2ycbcr(x_hat);
        x_hat = x_hat(:,:,1);
        im1 = rgb2ycbcr(HR(j).data);
        im_gt.out =im1(:, :, 1);
    end
    
    %% Compute the PSNR and SSIM values 
    [psnr_tvtv(j), ssim_tvtv(j)] = compute_diff(x_hat, im_gt.out, scaling_factor);
    [psnr_cnn(j), ssim_cnn(j)]   = compute_diff(im_w.out, im_gt.out, scaling_factor);

    % Print results
    if SHOW_RESULTS
        fprintf('Image %2d: TVTV - PSNR: %2.4f dB - SSIM: %2.4f \n', j, psnr_tvtv(j), ssim_tvtv(j))
        fprintf('        : CNN  - PSNR: %2.4f dB - SSIM: %2.4f \n', psnr_cnn(j) , ssim_cnn(j))
    end
    
end

% ======================================================
% Compute the mean over all images of PSNR and SSIM
psnr_tvtv_mean = mean(psnr_tvtv)
psnr_cnn_mean  = mean(psnr_cnn)
ssim_tvtv_mean = mean(ssim_tvtv)
ssim_cnn_mean  = mean(ssim_cnn)

%% Storing the results for each image and saving the results in a matrix 
save(filename_results, 'psnr_tvtv', 'psnr_cnn', 'ssim_tvtv', 'ssim_cnn',  ...
'psnr_tvtv_mean', 'psnr_cnn_mean', 'ssim_tvtv_mean', 'ssim_cnn_mean') 
% ======================================================
imshow(x_hat)