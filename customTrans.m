function[rXTr, rXTe] = customTrans (XTr, XTe, yTr, D)

%{
rXTr = XTr(:, [1:(D-1) D+1:end]);
rXTe = XTe(:, [1:(D-1) D+1:end]);
%}
%{
if (D ~= 0)
    NL = 49;
    rXTr = [XTr XTr(:,floor(D/NL)+1) .* XTr(:, mod(D, NL)+1)];
    rXTe = [XTe XTe(:,floor(D/NL)+1) .* XTe(:, mod(D, NL)+1)];
else
    rXTr = XTr;
    rXTe = XTe;
end
%}

if (D ~= 0)
    NL = 49;
    rXTr = [XTr  XTr(:, mod(D, NL)+1).^XTr(:,floor(D/NL)+1)];
    rXTe = [XTe  XTe(:, mod(D, NL)+1).^XTe(:,floor(D/NL)+1)];
else
    rXTr = XTr;
    rXTe = XTe;
end