# apm_ab — A/B test WebRTC APM cho ASR tổng đài

```bash
pip install livekit numpy scipy soundfile jiwer

# 1) Xử lý: WAV stereo (ch0 = khách, ch1 = bot) -> 8 biến thể + manifest.csv
python apm_ab.py process recordings/ -o apm_out --delay-ms auto

# 2) Chạy ASR của bạn trên các path trong apm_out/manifest.csv -> hyps.csv (file,variant,hyp)

# 3) Tính WER theo biến thể và nhóm SNR
python apm_ab.py wer --refs refs.csv --hyps hyps.csv --group-col snr_bucket --out wer_by_variant.csv
```

Biến thể: `raw, hpf, hpf_agc, hpf_ns, hpf_aec, hpf_aec_ns, hpf_aec_agc, full`
(chọn bớt bằng `--variants raw,hpf,hpf_aec`). File mono sẽ tự bỏ các biến thể có AEC.

Kết quả chạy thử trên cuộc gọi giả lập (8 kHz, vọng trễ 150 ms):

| Biến thể | Suy giảm vọng | Giọng khách lúc chỉ khách nói | Giọng khách lúc nói chen bot |
|---|---|---|---|
| hpf_aec | ~23 dB | giữ nguyên (−0.2 dB) | bị nén ~6 dB |
| hpf_aec_ns | ~35 dB | −0.7 dB | bị nén ~7 dB |

Độ trễ tự ước lượng (GCC-PHAT) ra đúng 150 ms.

Lưu ý: binding `livekit` chỉ bật/tắt module, không chỉnh mức NS/AGC. `echo_atten_db` trong manifest
là ERLE xấp xỉ, chỉ dùng để so sánh tương đối giữa các biến thể.
