trainingPrefix = '~/Documents/DTU/Semester 2/AI/Neural network/Project/trainingSet'
trainingMatrix = '~/Documents/DTU/Semester 2/AI/Neural network/Project/trainingMatrix'
for i =0:9
    
    currdir=[trainingPrefix '/' int2str(i)];
    disp(currdir);
    cd(currdir);
    clear dir
    list = dir('*.jpg');
    matrixDir = [trainingMatrix '/' int2str(i)];
    mkdir(matrixDir);
    len = length(list)
    count=0
    for j = 1:len
        nm=list(j).name;
        img = imread(nm);
        
        filename=[int2str(i) '_' int2str(count) '.txt'];
        count=count+1;
        imgWritePath = [matrixDir '/' filename]
        fid = fopen(imgWritePath, 'wt' );
        for ii=1:size(img,1)
            fprintf(fid,'%g\t',img(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid)
    end
end