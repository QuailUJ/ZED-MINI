# ZED MINI 動作錄製工作室

使用 ZED MINI 錄製彩色影片與人體骨架，完成 OpenSim Scale 與 IK，輸出個人化 OSIM、TRC 和 MOT。第一次使用請先閱讀 [完整使用教學](docs/USER_GUIDE.md)。

## 啟動

雙擊 `start_studio.bat` 即可只開啟主視窗，不會保留終端機；或在本資料夾執行：

```powershell
.\.venv\Scripts\python.exe zed_studio.py
```

使用目前 `.venv` 的 Python、ZED SDK、OpenSim、OpenCV、Tkinter 與 Pillow。
相機未連接時仍可閱讀內建教學。

已打包版本可直接執行 `ZED_MINI_Studio.exe`。EXE 採用資料夾模式，請保留旁邊的 `_internal` 目錄；錄製成果會建立在 EXE 同層的 `recordings`。

若要從原始碼重新打包：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean ZED_MINI_Studio.spec
```

完成後的程式位於 `dist/ZED_MINI_Studio/`。發佈時請壓縮或複製整個資料夾，不能只取出其中的 EXE。

第一次安裝建議使用 64 位元 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`pyzed` 不在 `requirements.txt` 內，需先安裝 [ZED SDK](https://www.stereolabs.com/developers/release/) 及其 Python API。
模型需要的 81 個幾何檔已精簡收錄於 `opensimPipeline/Geometry`，不需要另外下載整份 OpenSim 範例模型庫。

第一次啟動會顯示五頁使用教學：操作順序、相機準備、靜態校正、動作錄製與回放。
可按「先看主畫面」離開，隨時從「使用教學」重新開啟；「下次啟動仍顯示教學」決定是否下次自動顯示。
未接相機時錄製按鈕停用；連接失敗會提示原因，可再次連接，不影響教學。

## 錄製流程

1. 程式啟動後會自動連接相機。按「新增工作階段」輸入受測者或工作名稱；若要繼續以前的資料，按「開啟既有工作階段」。已完成校正且受測者與相機位置未改變時，可直接繼續錄動作。
2. 固定相機並保持水平，受測者面向相機，全身與雙腳完整入鏡。按下錄製後，程式會等待單一受測者的 20 個必要標記連續完整半秒，才正式開始保存。
3. 站穩後按「開始靜態校正錄製」，保持靜止數秒，再按「結束錄製並轉換」。程式會自動選取至少半秒、全身標記完整且最長的站姿區間。完成後可開始動作錄製。
4. 按「開始動作錄製」，做完動作後按「結束錄製並轉換」。不設 10 秒上限。動作中的內部缺點會強制線性補點並顯示品質警告，不會再因缺失時間較長而停止轉換。
5. 轉換完成後可繼續錄下一段。同一工作階段共用靜態模型；換人、移動相機或重新連接相機後需重新校正。
6. 本次程式開啟期間可選取錄製並回放、拖曳時間軸或切換骨架。正常關閉後只保留交付檔案。

勾選「啟用語音控制」後，可使用固定離線口令：「開始錄製」、「開始靜態錄製」和「結束錄製」。「開始錄製」會依目前是否已完成校正，自動選擇靜態或動作錄製；「結束錄製」只有正式開始錄影後才會執行，不會取消等待。辨識成功、正式開始錄影及結束錄影會播放不同提示音；語音辨識使用 Windows 內建繁體中文辨識器，不需連網。

MP4 為沒有骨架的左眼彩色影像（無音訊）。骨架於內建播放器疊加，切換不會改動影片。
本版只回放影片與骨架；OSIM／MOT 請使用 OpenSim 查看。

## 檔案與轉換

```text
recordings/01/
  01.osim
  01-errors.log                僅在發生錯誤時建立，所有錯誤依時間追加
  action_01/
    action_01.mp4
    action_01.trc
    action_01.mot
    action_01-notes.txt
  action_02/
    action_02.mp4
    action_02.trc
    action_02.mot
    action_02-notes.txt
```

實際流程是 `靜態 TRC + 基礎模型 → Scale → OSIM`，再用 `動作 TRC + OSIM → IK → MOT`。
程式執行期間會在隱藏的 `.studio` 目錄保存回放與 OpenSim 暫存資料；正常關閉時自動刪除。成功校正後會另外保留小型隱藏校正資料，供「開啟既有工作階段」繼續錄製。

- 靜態校正只使用完整站姿。動作錄製的內部缺點會強制線性補點，超過 0.2 秒會在完成訊息顯示品質警告；原始 TRC 不會被修改。
- 走入與走出畫面造成的首尾缺失會從處理後 TRC 裁掉。有效區段內的缺口會強制線性補點；每段動作的備註檔會揭露補點數量、時間及 IK 誤差。空資料或不遞增時間仍會停止轉換。
- 擷取時間來自 ZED 影格時間戳記；原始 TRC 的 DataRate 記錄實際平均取樣率，CameraRate 記錄相機幀率，計算使用每列時間。
- 轉換後檢查模型可載入、MOT 幀數／時間／有限數值與標記誤差報告，通過後才顯示完成。
- 正面錄影的 OpenSim 前後屈曲方向會自動修正，並以腰椎角度抵消骨盆傾斜，使軀幹維持直立。備註檔會標示此處理；IK 誤差代表修正前的標記擬合結果。
- 本次程式開啟期間可按「重新轉換選取錄製」。關閉後暫存資料會清除，無法再從介面重新轉換。
- 相機中斷或關閉視窗時保存已收到的資料；關閉期間等待進行中的 OpenSim 轉換完成。突然斷電／強制終止不保證 MP4 可播放。
- 轉換久候可按「取消轉換」，停止該次子程序並標示已取消，原始影片與 TRC 不變；也可在等待關閉時取消。取消產生的部分結果不能視為完成。
- 回放資料載入與影片解碼在背景執行，正常播放連續讀幀，拖曳時只處理最新位置。損壞檔案會顯示錯誤，不會讓整個介面停止回應。

沿用原本 20 個標記、模型權重與模板質量 **75.337 kg**。本版為運動學分析，沒有新增動力學分析。
地面對齊只做平移，不會校正相機傾斜。IK 標記誤差供檢查，不等於實際關節角度準確度已驗證。

## 命令列相容

```powershell
.\.venv\Scripts\python.exe capture_to_trc.py recordings\new_trial.trc --duration 10
.\.venv\Scripts\python.exe run_pipeline.py recordings\static_trial.trc recordings\new_trial.trc
```

命令列擷取仍預設 10 秒，空白鍵開始、Q 提前結束，且只輸出 TRC。錄製 MP4 請使用工作室。
轉換程式不再詢問資料夾名稱，改在動作 TRC 旁建立唯一 `conversion_*` 資料夾，包含 static 與 motion 結果。
相對輸入路徑仍以專案目錄為基準。OpenSim XML 使用相對路徑，因此轉換輸出需與專案位於同一磁碟。

## 驗證

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_studio tests.test_studio_ui -v
```

測試涵蓋短缺失、頭尾／長缺失、無效時間、人物鎖定、MP4 時長、檔案不覆蓋、模擬相機中斷、連續錄製與失敗報告。
另有無相機 GUI 測試：教學進退與偏好保存、無設備連線重試、回放定位／切換／關閉、損壞骨架、連續解碼及取消轉換。
使用 0.4 秒的人為解碼延遲驗證介面事件仍可處理；這不等於實機錄製效能已驗證。
合成 Scale → IK 在 `test_data/studio_verification_*` 下另存結果，不改動既有錄製。

舊版日期命名的實驗程式只保留在本機 `legacy_scripts/`，不納入 GitHub；目前 GUI 入口是 `start_studio.bat`。環境與 TRC 檢查工具位於 `tools/`，測試位於 `tests/`。

2026-10-04 已用 ZED MINI 實際完成靜態校正、動作錄製、Scale、IK、影片回放及 OpenSim 姿勢比對。IK 使用左右髖與骨盆中心三點估計逐幀骨盆方向；輸出 MOT 另套用已確認的正面錄影方向與直立軀幹修正。新錄製資料夾及同名成果使用 `action_01` 格式，既有成果不自動改名。
