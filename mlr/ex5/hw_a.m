clear;
clc;

files = dir("yalefaces");
X = [];
for i = 3:rows(files)
  name = files(i).name;
  filename = strcat("yalefaces/",name);
  img = imread(filename)(:);
  X = [X, img];
end

