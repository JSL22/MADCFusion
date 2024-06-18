
% Our = ["MADCFusion", "A1", "A2"];
% nestfusion = ["nestfusion", "C1", "C2"];   % avg
% FusionGAN = ["FusionGAN", "D1", "D2"];
% GTF = ["GTF", "E1", "E2"];
% IFEVIP = ["IFEVIP", "F1", "F2"];
% IFCNN = ["IFCNN", "G1", "G2"];
% CBF = ["CBF", "H1", "H2"];
% MFEIF = ["MFEIF", "I1", "I2"];


% RoadScene数据集检测数据
% ir_path = '.\dataset\RoadScene\ir';
% vis_path = '.\dataset\RoadScene\vis';
% fusion_path = '.\fusion\RoadScene\';
% benchmark_file = '.\metric_excel\RoadScene1.xls';


% 测试
% ["融合图像的文件夹名", "excel表头"，"excel表体"]
Our = ["MADCFusion", "A1", "A2"]; 
ir_path = '.\dataset\TNO\ir';
vis_path = '.\dataset\TNO\vis'; 
fusion_path = '.\fusion_images\TNO\';
benchmark_file = '.\metric_results\TNO1.xls';
tip = 1;     % 计算指标， metrics ： 1， metrics2：2

parameter = Our;

% 源图像后缀
img_suffix = ["*.jpg", "*.png", "*.bmp"];
fusion_names = [];

% 查看融合图片的个数和文件名
for i = 1:length(img_suffix)
    fusion_images = dir(fullfile(fusion_path, parameter(1), "\", img_suffix(i)));
    fusion_names = horzcat(fusion_names, {fusion_images.name});
end
img_nums = length(fusion_names);

% 融合指标结果
EN_set = zeros(1, img_nums);
SF_set = EN_set; SD_set = EN_set; PSNR_set = EN_set; 
MSE_set = EN_set; MI_set = EN_set; VIF_set = EN_set;
AG_set = EN_set; CC_set = EN_set; SCD_set = EN_set;
Qabf_set = EN_set; Nabf_set = EN_set; SSIM_set = EN_set;
MS_SSIM_set = EN_set; FMI_pixel_set = EN_set;
FMI_dct_set = EN_set; FMI_w_set = EN_set;

for i = 1:img_nums
   ir_img = imread(fullfile(ir_path, fusion_names{i}));
   vis_img = imread(fullfile(vis_path, fusion_names{i}));
   fusion_img = imread(fullfile(fusion_path, parameter(1), "\", fusion_names{i}));
   fprintf('第%d张待计算图像：%s\n',i, fusion_names{i})
   % 设置为灰度图像 if ndims(img) > 2
   if length(size(ir_img)) > 2
       ir_img = rgb2gray(ir_img);
   end
   if length(size(vis_img)) > 2
       vis_img = rgb2gray(vis_img);
   end
   if length(size(fusion_img)) > 2
       fusion_img = rgb2gray(fusion_img);
   end
   
   % 计算指标， metrics ： 1， metrics2：2
   [EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w] ...
       = analysis_Reference(fusion_img, ir_img, vis_img, tip);
   
   EN_set(i) = EN; SF_set(i) = SF; SD_set(i) = SD; PSNR_set(i) = PSNR; 
   MSE_set(i) = MSE; MI_set(i) = MI; VIF_set(i) = VIF; 
   AG_set(i) = AG; CC_set(i) = CC; SCD_set(i) = SCD; 
   Qabf_set(i) = Qabf; Nabf_set(i) = Nabf; SSIM_set(i) = SSIM;
   MS_SSIM_set(i) = MS_SSIM; FMI_pixel_set(i) = FMI_pixel;
   FMI_dct_set(i) = FMI_dct; FMI_w_set(i) = FMI_w;
   
end

% 指标结果写入文件中
% 一列一个方法的测量数据
local_head = parameter(2);
local_body = parameter(3);
method_name = cellstr(parameter(1));

%%%%%%%%%%%%%%%% metrics %%%%%%%%%%%%%%%%%
if tip == 1
    metrics = ["EN", "SF", "SD", "PSNR", "MSE", "MI", "VIF", "AG", "CC", "SCD", "Qabf"];
    metrics_data = [EN_set; SF_set; SD_set; PSNR_set; MSE_set; MI_set; VIF_set; AG_set; CC_set; SCD_set; Qabf_set;];
    [r, c] = size(metrics_data);

    % 表头方法名
    for i = 1:length(metrics)
        xlswrite(benchmark_file, method_name, metrics(i) ,local_head)
    end

    %表体指标数据
    for i = 1:r
        xlswrite(benchmark_file, metrics_data(i, :)', metrics(i) ,local_body)
    end
end

%%%%%%%%%%%%%%%% metrics2 %%%%%%%%%%%%%%%%%
if tip == 2
    metrics2 = ["Nabf", "SSIM", "MS_SSIM", "FMI_pixel", "FMI_dct", "FMI_w"];
    metrics2_data = [Nabf_set; SSIM_set; MS_SSIM_set; FMI_pixel_set; FMI_dct_set; FMI_w_set];
    [r, c] = size(metrics2_data);

    % 表头方法名
    for i = 1:length(metrics2)
        xlswrite(benchmark_file, method_name, metrics2(i) ,local_head)
    end

    %表体指标数据
    for i = 1:r
        xlswrite(benchmark_file, metrics2_data(i, :)', metrics2(i) ,local_body)
    end
end

