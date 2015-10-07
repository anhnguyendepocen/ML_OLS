%gets the interactions that are the most correlated with y => useless
function[rXTr, rXTe] = XSortLR (XTr, XTe, yTr, D)

N = size(XTr, 2);
%creates the list of all crossed variables
corrList = zeros(22*22,3);
k = 1;
for i=2:23
   for j=2:23
       if (i ~= j)
           corrList(k,:) = [abs(corr(yTr, XTr(:,i).*XTr(:,j))) i j];
           k = k+1;
       end
   end
end

N = 23;

rXTr = [XTr zeros(length(XTr), D)];
rXTe = [XTe zeros(length(XTe), D)];

corrList = flipud(sortrows(corrList, 1));
for i=1:D
    rXTr(:, N+i) = XTr(:, corrList(i, 2)) .* XTr(:, corrList(i, 3));
    rXTe(:, N+i) = XTe(:, corrList(i, 2)) .* XTe(:, corrList(i, 3));
end