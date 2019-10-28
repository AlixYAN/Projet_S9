
function [res] = tempo_stem_file(filename)

%%%%%%%%% OSS
[wav_data, wav_sr] = audioread(filename);
aInfo = audioinfo(filename);
bps = aInfo.BitsPerSample;

wav_data = wav_data * 32767.0 / 32768.0;

[oss, oss_sr] = onset_signal_strength(wav_data, wav_sr);

%%%%%%%%% BH
bh_cands = beat_histogram(oss, oss_sr);

bpm_min = 40;
bpm_max = 200;
res = zeros(1,(bpm_max-bpm_min));
edges = bpm_min:1:bpm_max;

for i=1:size(bh_cands,1)
    a = histcounts(bh_cands(i,:),edges);
    b = a/size(bh_cands,2);
    res = res + b;
end


end


