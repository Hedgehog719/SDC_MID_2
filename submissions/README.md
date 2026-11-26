# 提交檔

只保留最終選中、private 最好，以及報告裡比較用到的關鍵版本。分數為 Kaggle 的 public / private。

## Track01（`track1/`）

| 檔案 | public | private | 說明 |
|---|---|---|---|
| `t1_rs26_dt35.csv` | 0.069 | **0.065** | 純影像 BA＋自校正鏡頭＋移動物體遮罩＋滾動快門 26 ms＋時間平移 −35 ms；private 最好，但當時沒選 |
| `t1_fuse5050_c10gcam3_fullvio_v4pure.csv` | 0.066 | 0.107 | 最終選中：位姿圖與 VIO 的 50/50 融合 |
| `t1_fuse_c10_gcam3.csv` | 0.065 | 0.126 | public 最好的位姿圖融合 |
| `t1_sfm_rs34_dt35_late.csv` | 0.069 | 0.066 | 賽後晚交：同一套純影像 BA，滾動快門 34 ms、時間差 −35 ms 都用物理推算；重現指令見主 README |

## Track02（`track2/`）

| 檔案 | public | private | 說明 |
|---|---|---|---|
| `t2_v1.csv` | 4.227 | 4.047 | 第一版前端 |
| `t2_pcm_sfmx_c.csv` | 0.119 | 1.407 | 跨趟約束寫進 BA |
| `t2_sfmx_c_rn1d_dt34.csv` | 0.061 | 1.416 | ＋路面法向＋時間差 |
| `t2_sfmx_static_far_rs34_dt41.csv` | 0.045 | 1.263 | ＋靜止車輛、遠景、滾動快門（沒有虛線約束） |
| `t2_fuse5050_nd34_sfu34_dt41.csv` | 0.037 | 0.664 | public 最好的融合版 |
| `t2_sfmx_ult_r0_dt34.csv` | 0.064 | 0.305 | 近距虛線寫進 BA |
| `t2_sfmx_homo_ult_frp_dt41.csv` | 0.040 | 0.184 | ＋柏油路面量尺 |
| `t2_sfmx_homo_ult_frp_v2_dt41.csv` | 0.041 | **0.187** | 最終選中（可完整重現的版本），第 1 名 |
| `t2_sfmx_homo_ult_frp_v2_dt35_late.csv` | 0.046 | 0.183 | 賽後晚交：同一個 BA，時間差改用純物理推算的 −35 ms |
