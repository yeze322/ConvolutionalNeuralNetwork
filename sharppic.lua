require 'image'
cmd = torch:CmdLine();
cmd:option('--file', '1.jpg','input file name')
cmd:option('--th',150,'sharpen thresold');
cmd:option('--square',160,'square size')
cmd:option('--factor',0.8,'threasold factor')
cmd:option('--globalfac','0.1', 'global bias factor')
opt = cmd:parse(arg or {});

img = image.load(opt.file);
--img:mul(255):div(opt.th);
--img = img:int();

--TH = opt.th/255
for row = 1,img:size(2) do
    for col = 1,img:size(3) do
        local mm = img[{{},row,col}]:min()
        --[[
        for channel=1,img:size(1) do
            if img[channel][row][col] < TH
            then
                img[channel][row][col] = 0
            else img[channel][row][col] = 1
            end
        end
        ]]--
        img[1][row][col] = mm;
    end
end
MEAN = img:mean()*255;

function sharpen255pic(svimg, a, b, c)
    svimg = svimg:mul(255):int()
    --static stage
    rowdiv = svimg:size(1)/opt.square;
    coldiv = svimg:size(2)/opt.square;
    
    avgtensot = torch.Tensor(rowdiv,coldiv);
    for i=1,rowdiv do
        for j=1,coldiv do
            local area = svimg[{{(i-1)*opt.square+1, i*opt.square},{(j-1)*opt.square+1,j*opt.square}}]:double();
            local x = area:mean();
            local thre = a * x * x / 255 + b * x + MEAN * c;
            print(x .. ' ' .. thre)
            area:div(thre);
            area = area:int();
            svimg[{{(i-1)*opt.square+1, i*opt.square},{(j-1)*opt.square+1,j*opt.square}}] = area;
        end
    end
    
    image.save(a .. '_' .. b .. '.jpg',svimg)
end

for a=0.5,0.6,0.05 do
    for b=0.5,0.6,0.05 do
        local svimg = torch.Tensor(img[1]:size());
        svimg:copy(img[1]);
        sharpen255pic(svimg,a,b,0.0)
    end
end
