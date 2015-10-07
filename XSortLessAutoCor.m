function[rXTr, rXTe] = XSortLessAutoCor (XTr, XTe, yTr, D)

N = 37;
%creates the list of all crossed variables

tmpX = [XTr zeros(length(XTr), 39*13)];

for i=2:36
   for j=37:49
       if (i ~= j)
           tmpX(:, size(XTr, 2)+(i-2)*13+(j-36)) = XTr(:,i) .* XTr(:, j);
       end
   end
end
correl = abs(corr(tmpX));
k = 2;

k = 1;
for i=2:36
   for j=37:49
       if (i ~= j)
           corrList(k,:) = [max(correl([2:N-1 50:k-1 k+1:end], k)) i j];
           k = k+1;
       end
   end
end

rXTr = [XTr(:, 1:N) zeros(length(XTr), D)];
rXTe = [XTe(:, 1:N) zeros(length(XTe), D)];

%corrList = flipud(sortrows(corrList, 1));
corrList = sortrows(corrList, 1);
for i=1:D
    rXTr(:, N+i) = XTr(:, corrList(i, 2)) .* XTr(:, corrList(i, 3));
    rXTe(:, N+i) = XTe(:, corrList(i, 2)) .* XTe(:, corrList(i, 3));
end