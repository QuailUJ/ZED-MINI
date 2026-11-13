# ZED MINI 動作錄製工作室

Windows 使用者可從 [GitHub Releases](https://github.com/QuailUJ/ZED-MINI/releases) 下載完整 ZIP。從原始碼啟動請使用 `start_studio.bat`。拍攝前可參考 [拍攝環境說明](docs/RECORDING_SETUP.html)。

座標旋轉、度／弧度及耦合座標已用 OpenSim 對照。2026-11-13 精簡後，既有 158 幀 MOT 與處理後 TRC 均與修改前逐位元一致。程式會攔截超出模型可達距離的踝與腳跟組合，並標記相鄰幀的大角度跳變。新版實機錄製仍待驗證。

使用 ZED MINI 錄製彩色影片與人體骨架，完成 OpenSim Scale 與 IK，輸出個人化 OSIM、TRC 和 MOT。第一次使用請先閱讀 [完整使用教學](docs/USER_GUIDE.md)。

## 啟動

雙擊 `start_studio.bat` 即可只開啟主視窗，不會保留終端機；或在本資料夾執行：

```powershell
.\.venv\Scripts\python.exe zed_studio.py
```

使用目前 `.venv` 的 Python、ZED SDK、OpenSim、OpenCV、Tkinter 與 Pillow。
相機未連接時仍可閱讀內建教學。

已打包版本可直接執行 `ZED_MINI_Studio.exe`。EXE 採用資料夾模式，請保留旁邊的 `_internal` 目錄；錄製成果會建立在 EXE 同層的 `recordings`。目前發佈版對應 **ZED SDK 5.5.0**，會自動使用電腦已安裝的 ZED、NVIDIA、CUDA 與 TensorRT 元件，不必手動輸入安裝路徑。

若一般版本啟動後自行關閉，請改開 `ZED_MINI_Studio_SafeMode.exe`。安全模式不會自動初始化相機或語音，方便確認問題是否來自 ZED 人體追蹤；之後可手動按「連接相機」。診斷過程會寫入同層的 `studio_diagnostics.log`。

若要從原始碼重新打包，先將 StereoLabs 的 64 位元 Python 3.11 wheel `pyzed-5.5-cp311-cp311-win_amd64.whl` 放在專案的 `.analysis` 內。SDK 安裝目錄中的 `get_python_api.py` 會依 SDK 和 Python 版本取得相符的套件。[StereoLabs Python API 安裝說明](https://docs.stereolabs.com/docs/development/api-languages/python)。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools\prepare_zed55_build.py
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean ZED_MINI_Studio.spec
```

完成後的程式位於 `dist/ZED_MINI_Studio/`。發佈時請壓縮或複製整個資料夾，不能只取出其中的 EXE。

第一次安裝建議使用 64 位元 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`pyzed` 不在 `requirements.txt` 內，需先安裝 [ZED SDK 5.5.0](https://www.stereolabs.com/developers/release/) 及其 Python API。程式預設從 ZED 安裝程式建立的 `ZED_SDK_ROOT_DIR` 取得原生元件；若環境變數不存在，才使用官方預設位置 `C:\Program Files (x86)\ZED SDK`。
模型需要的 81 個幾何檔已精簡收錄於 `opensimPipeline/Geometry`，不需要另外下載整份 OpenSim 範例模型庫。

第一次啟動會顯示五頁使用教學：操作順序、相機準備、靜態校正、動作錄製與回放。
可按「先看主畫面」離開，隨時從「使用教學」重新開啟；「下次啟動仍顯示教學」決定是否下次自動顯示。
未接相機時錄製按鈕停用；連接失敗會提示原因，可再次連接，不影響教學。

## 錄製流程

1. 程式啟動後會自動連接相機。按「新增工作階段」輸入受測者或工作名稱；若要繼續以前的資料，按「開啟既有工作階段」。已完成校正且受測者與相機位置未改變時，可直接繼續錄動作。
2. 固定相機並保持水平，受測者面向相機，全身與雙腳完整入鏡。按下錄製後，程式會等待單一受測者的 20 個必要標記連續完整半秒，才正式開始保存。
3. 站穩後按「開始靜態校正錄製」，保持靜止數秒，再按「結束錄製並轉換」。程式會自動選取至少半秒、全身標記完整且最長的站姿區間。完成後可開始動作錄製。
4. 按「開始動作錄製」，做完動作後按「結束錄製並轉換」。動作標記的內部缺口以兩端有效樣本線性補點，超過 0.2 秒會記錄品質警告。相鄰擷取時間中斷超過 0.2 秒或腳點超出模型可達距離時，會停止轉換並顯示原因。
5. 轉換完成後可繼續錄下一段。同一工作階段共用靜態模型；換人、移動相機或重新連接相機後需重新校正。
6. 本次程式開啟期間可選取錄製並回放、拖曳時間軸或切換骨架。關閉後可直接開啟輸出的 MP4，原始骨架與轉換結果另存於 `diagnostics`。

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
程式在 `.studio` 保存回放與 OpenSim 暫存。正常關閉時，會先將原始 TRC、骨架、轉換結果與所用模型封存到 `diagnostics`，再清理暫存。動作影片若尚未輸出，也會封存。成功校正後的隱藏校正資料供下次開啟工作階段使用。

- 靜態校正只使用完整站姿。動作錄製的內部缺點會強制線性補點，超過 0.2 秒會在完成訊息顯示品質警告；原始 TRC 不會被修改。
- 走入與走出畫面造成的首尾缺失會從處理後 TRC 裁掉。有效區段內的缺口會強制線性補點；每段動作的備註檔會揭露補點數量、時間及 IK 誤差。空資料或不遞增時間仍會停止轉換。
- 擷取時間來自 ZED 影格時間戳記；原始 TRC 的 DataRate 記錄實際平均取樣率，CameraRate 記錄相機幀率，計算使用每列時間。
- 轉換後檢查模型可載入、MOT 幀數／時間／有限數值與標記誤差報告，通過後才顯示完成。
- IK 以左右髖與骨盆中心的逐幀方向輔助估計骨盆傾角。MOT 保留求解器的角度正負號及腰椎活動。角度連續性警告記錄座標與時間，供回放比對。
- 每段動作以第一個有效影格的骨盆標記 X/Z 與最低腳部標記 Y 建立原點。OpenSim 求解後，再以第一幀的骨盆位置與縮放後足部網格計算固定平移，讓第一幀骨盆 X/Z 歸零、最低足部表面位於 Y=0.005 m。交付 TRC 也套用同一平移，以便和 MOT 疊圖。
- 地面定位只平移整段動作，保留每幀之間的垂直位移。從腳接觸支撐面的姿勢開始錄製。後續影格若穿入地板，notes 會記錄最低足部表面高度。驗證範圍涵蓋 MOT 的取樣影格。
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
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py'
```

測試涵蓋短缺失、頭尾／長缺失、無效時間、人物鎖定、MP4 時長、檔案不覆蓋、模擬相機中斷、連續錄製與失敗報告。
另有無相機 GUI 測試：教學進退與偏好保存、無設備連線重試、回放定位／切換／關閉、損壞骨架、連續解碼及取消轉換。
使用 0.4 秒的人為解碼延遲驗證介面事件仍可處理；這不等於實機錄製效能已驗證。
合成 Scale → IK 在 `test_data/studio_verification_*` 下另存結果，不改動既有錄製。

舊版日期命名的實驗程式只保留在本機 `legacy_scripts/`，不納入 GitHub；目前 GUI 入口是 `start_studio.bat`。環境與 TRC 檢查工具位於 `tools/`，測試位於 `tests/`。

2026-10-04 曾用 ZED MINI 實際完成靜態校正、動作錄製、Scale、IK、影片回放及 OpenSim 姿勢比對。現在的程式修訂使用既有錄製與模擬測試驗證。新錄製資料夾及同名成果使用 `action_01` 格式。
