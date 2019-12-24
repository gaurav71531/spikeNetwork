function plotSpikeCascaseVert(nspike)

[nNeuron,N] = size(nspike);
lineWidth = 1.5;

figure;
strAll = '[';
for i = 1:nNeuron
    str = sprintf('ax%d = axes;', i);
    strNext = sprintf('ax%d;', i);
    strAll = [strAll, strNext];
    eval(str);
end
strAll(end) = ']';
eval(['linkaxes(' strAll ');']);
for i = 1:nNeuron
    str = sprintf('h%d = stem(ax%d, 1:N, nspike(%d,:)+%d);', i,i,i,i-1);
    eval(str);
    str = sprintf('set(h%d, ''marker'', ''none'', ''BaseValue'', %d);', i,i-1);
    eval(str);
    str = sprintf('set(h%d, ''LineWidth'', %d);', i,lineWidth);
    eval(str);
    str = sprintf('set(h%d, ''color'', ''k'');', i);
    eval(str);
end

axis(ax1, [1,N,0,nNeuron]);
for i = 2:nNeuron
    str = sprintf('set(ax%d, ''color'', ''none'', ''xtick'', '''', ''ytick'', '''');', i);
    eval(str);
end
ax1.YTick = 0.5:1:nNeuron-0.5;
neuronLabel = num2cell(1:nNeuron);
neuronLabel = cellfun(@num2str, neuronLabel, 'UniformOutput', false);
ax1.YTickLabel = neuronLabel;
