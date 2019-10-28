clear
close all
clc

data = importfile_data('../../data.csv')

varTypes = {'double', 'double', 'double', 'double', 'double', 'double'};
varNames = {'A0','A1','RA','P1','P2','SUM'};
data2 = table('Size',[size(data,1),6],'VariableTypes',varTypes,'VariableNames',varNames);
label_vector = data(:,end);
data = data(:,1:end-1);
data = [data data2 label_vector];

for i=1:size(data,1)
filename = ['../../dataset/',char(table2array(label_vector(i,1))),'/',char(table2array(data(i,1)))];
Beat_hist = tempo_stem_file(filename);
[pks,locs] = findpeaks(Beat_hist);
[A,P] = maxk(pks,2);
SUM = sum(Beat_hist);

data(i,15) = num2cell(A(1)/SUM);
data(i,16) = num2cell(A(2)/SUM);
data(i,17) = num2cell(A(2)/A(1));
data(i,18) = num2cell(locs(P(1))+40);
data(i,19) = num2cell(locs(P(2))+40);
data(i,20) = num2cell(SUM);

end