function[rXTr, rXTe] = customTransLR (XTr, XTe, yTr, D)

rXTr = XTr;
rXTe = XTe;

if (D ~= 0)
    
    ds = [169 191 384 205 491 120 131 373 183 513 292 380 182 490 35 277];
    NL = 23;
    for d=1:D 
      rXTr = [rXTr XTr(:,floor(ds(d)/NL)+1) .* XTr(:, mod(ds(d), NL)+1)];
      rXTe = [rXTe XTe(:,floor(ds(d)/NL)+1) .* XTe(:, mod(ds(d), NL)+1)];

    end
    
    %{
    NL = 23;
    rXTr = [XTr XTr(:,floor(D/NL)+1) .* XTr(:, mod(D, NL)+1)];
    rXTe = [XTe XTe(:,floor(D/NL)+1) .* XTe(:, mod(D, NL)+1)];
    %}
    %{
    rXTr = XTr(:, [1:D-1 D+1:end]);
    rXTe = XTe(:, [1:D-1 D+1:end]);
    %}
else
    rXTr = XTr;
    rXTe = XTe;
end