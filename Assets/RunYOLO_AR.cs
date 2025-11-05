using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.InferenceEngine;

public class RunYOLO_AR : MonoBehaviour
{
    // ====================================================================
    // 1. AR + MODEL FIELDS
    // ====================================================================

    [Header("AR")]
    [SerializeField] private ARCameraManager cameraManager;

    [Header("Model")]
    [SerializeField] private ModelAsset yoloModelAsset;
    [SerializeField] private TextAsset labelsFile;
    [SerializeField] private int modelInputWidth = 640;
    [SerializeField] private int modelInputHeight = 640;

    [Header("Detection Settings")]
    [SerializeField] [Range(0.1f, 0.9f)] private float confidenceThreshold = 0.25f;
    [SerializeField] [Range(0.1f, 0.9f)] private float nmsThreshold = 0.45f;
    [SerializeField] [Range(0.01f, 1f)] private float inferenceInterval = 0.1f;
    [SerializeField] private int maxDetections = 50;

    private const BackendType BACKEND = BackendType.GPUCompute;
    private Model runtimeModel;
    private Worker worker;

    private string[] labels;
    private int numClasses = 80;

    private Texture2D cpuTexture;
    private Texture2D modelInputTexture;

    private bool isRunning = false;
    private float lastInferenceTime;

    // ====================================================================
    // 2. UI OVERLAY / BOX DRAWING
    // ====================================================================

    [Header("UI Overlay")]
    [SerializeField] private Canvas overlayCanvas;
    [SerializeField] private RectTransform overlayRoot;
    [SerializeField] private Color boxColor = Color.yellow;
    [SerializeField] private float boxLineWidth = 3f;
    [SerializeField] private int labelFontSize = 16;
    [SerializeField] private RawImage debugImage;

    [Serializable]
    public struct BoundingBox
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public float confidence;
        public int classIndex;
        public string label;

        public float centerX => x + width * 0.5f;
        public float centerY => y + height * 0.5f;
        public float X2 => x + width;
        public float Y2 => y + height;
    }

    private readonly List<BoundingBox> currentDetections = new List<BoundingBox>();
    private readonly List<BoundingBoxUI> boxUIPool = new List<BoundingBoxUI>();

    // Simple UI box representation
    private class BoundingBoxUI
    {
        public GameObject rootObject;
        public RectTransform rectTransform;
        public RawImage[] lines; // 4 lines for border
        public Text label;
    }

    // ====================================================================
    // 3. UNITY LIFECYCLE
    // ====================================================================

    void Awake()
    {
        LogDebug("RunYOLO_AR: Awake()");

        InitializeLabels();
        InitializeModel();
    }

    void OnEnable()
    {
        LogDebug("RunYOLO_AR: OnEnable");
        if (cameraManager != null)
        {
            cameraManager.frameReceived += OnCameraFrameReceived;
        }
        else
        {
            Debug.LogError("RunYOLO_AR: ARCameraManager is not assigned!");
        }
    }

    void Start()
    {
        LogDebug("RunYOLO_AR: Start");
        
        // Validate overlay setup
        if (overlayRoot == null)
        {
            Debug.LogError("RunYOLO_AR: overlayRoot not assigned! Bounding boxes will not display.");
        }
        
        if (overlayCanvas == null)
        {
            Debug.LogWarning("RunYOLO_AR: overlayCanvas not assigned, trying to find it...");
            overlayCanvas = overlayRoot?.GetComponentInParent<Canvas>();
        }

        PreallocateBoxPool(maxDetections);
    }

    void OnDisable()
    {
        if (cameraManager != null)
        {
            cameraManager.frameReceived -= OnCameraFrameReceived;
        }
    }

    void OnDestroy()
    {
        CleanupResources();
    }

    // ====================================================================
    // 4. INITIALIZATION
    // ====================================================================

    private void InitializeLabels()
    {
        if (labelsFile != null)
        {
            labels = labelsFile.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            if (labels.Length > 0)
            {
                numClasses = labels.Length;
            }
            LogDebug($"RunYOLO_AR: Loaded {numClasses} labels.");
        }
        else
        {
            numClasses = 80;
            labels = new string[numClasses];
            for (int i = 0; i < numClasses; i++)
            {
                labels[i] = $"cls_{i}";
            }
            Debug.LogWarning("RunYOLO_AR: labelsFile not set, using dummy labels.");
        }
    }

    private void InitializeModel()
    {
        if (yoloModelAsset != null)
        {
            try
            {
                runtimeModel = ModelLoader.Load(yoloModelAsset);
                worker = new Worker(runtimeModel, BACKEND);
                LogDebug("RunYOLO_AR: Sentis/Barracuda worker initialized.");
            }
            catch (Exception e)
            {
                Debug.LogError($"RunYOLO_AR: Failed to initialize model: {e.Message}");
            }
        }
        else
        {
            Debug.LogError("RunYOLO_AR: No YOLO model asset assigned!");
        }
    }

    private void PreallocateBoxPool(int count)
    {
        if (overlayRoot == null)
        {
            Debug.LogError("RunYOLO_AR: Cannot preallocate boxes - overlayRoot is null!");
            return;
        }

        for (int i = 0; i < count; i++)
        {
            var boxUI = CreateBoundingBoxUI();
            boxUI.rootObject.SetActive(false);
            boxUIPool.Add(boxUI);
        }
        LogDebug($"RunYOLO_AR: Preallocated {count} UI boxes.");
    }

    private void CleanupResources()
    {
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        if (cpuTexture != null)
        {
            Destroy(cpuTexture);
            cpuTexture = null;
        }

        if (modelInputTexture != null)
        {
            Destroy(modelInputTexture);
            modelInputTexture = null;
        }

        foreach (var boxUI in boxUIPool)
        {
            if (boxUI.rootObject != null)
                Destroy(boxUI.rootObject);
        }
        boxUIPool.Clear();
    }

    // ====================================================================
    // 5. AR FRAME CALLBACK
    // ====================================================================

    void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (worker == null || isRunning)
            return;

        // Throttle inference based on interval
        if (Time.time - lastInferenceTime < inferenceInterval)
            return;

        if (!cameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
            return;

        lastInferenceTime = Time.time;
        LogDebug("RunYOLO_AR: OnCameraFrameReceived - processing frame");
        StartCoroutine(RunYoloOnCurrentFrame(cpuImage));
    }

    // ====================================================================
    // 6. PROCESS FRAME + RUN YOLO
    // ====================================================================

    IEnumerator RunYoloOnCurrentFrame(XRCpuImage cpuImage)
    {
        isRunning = true;

        try
        {
            using (cpuImage)
            {
                LogDebug($"RunYOLO_AR: Got CPU image {cpuImage.width}x{cpuImage.height}");

                // Convert XRCpuImage to Texture2D
                ConvertCpuImageToTexture(cpuImage);

                // Resize to model input size
                ResizeTexture();

                // Optional debug display
                if (debugImage != null)
                {
                    debugImage.texture = modelInputTexture;
                }

                // Run inference
                RunInference();

                // Get and parse output
                ParseModelOutput();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"RunYOLO_AR: Error processing frame: {e.Message}\n{e.StackTrace}");
        }
        finally
        {
            isRunning = false;
        }

        yield return null;
    }

    private void ConvertCpuImageToTexture(XRCpuImage cpuImage)
    {
        if (cpuTexture == null || 
            cpuTexture.width != cpuImage.width || 
            cpuTexture.height != cpuImage.height)
        {
            if (cpuTexture != null)
                Destroy(cpuTexture);
                
            cpuTexture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGBA32, false);
        }

        var conversionParams = new XRCpuImage.ConversionParams(
            cpuImage,
            TextureFormat.RGBA32,
            XRCpuImage.Transformation.MirrorX
        );

        var rawTextureData = cpuTexture.GetRawTextureData<byte>();
        cpuImage.Convert(conversionParams, rawTextureData);
        cpuTexture.Apply();
    }

    private void ResizeTexture()
    {
        if (modelInputTexture == null ||
            modelInputTexture.width != modelInputWidth ||
            modelInputTexture.height != modelInputHeight)
        {
            if (modelInputTexture != null)
                Destroy(modelInputTexture);
                
            modelInputTexture = new Texture2D(modelInputWidth, modelInputHeight, TextureFormat.RGBA32, false);
        }

        ResizeInto(cpuTexture, modelInputTexture);
    }

    private void ResizeInto(Texture2D source, Texture2D target)
    {
        RenderTexture rt = null;
        RenderTexture prev = RenderTexture.active;

        try
        {
            rt = RenderTexture.GetTemporary(target.width, target.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(source, rt);
            RenderTexture.active = rt;
            target.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0);
            target.Apply();
        }
        finally
        {
            RenderTexture.active = prev;
            if (rt != null)
            {
                RenderTexture.ReleaseTemporary(rt);
            }
        }
    }

    private void RunInference()
    {
        LogDebug("RunYOLO_AR: Running YOLO inference.");

        using (var inputTensor = new Tensor<float>(new TensorShape(1, 3, modelInputHeight, modelInputWidth)))
        {
            TextureConverter.ToTensor(modelInputTexture, inputTensor, default(TextureTransform));
            worker.Schedule(inputTensor);
        }
    }

    private void ParseModelOutput()
    {
        var gpuOutput = worker.PeekOutput() as Tensor<float>;

        if (gpuOutput == null)
        {
            Debug.LogWarning("RunYOLO_AR: YOLO output tensor is null or not float.");
            return;
        }

        var shape = gpuOutput.shape;
        LogDebug($"RunYOLO_AR: YOLO output shape rank={shape.rank}, length={shape.length}");

        using (var cpuOutput = gpuOutput.ReadbackAndClone() as Tensor<float>)
        {
            ParseYoloV8Output(cpuOutput);
        }
    }

    // ====================================================================
    // 7. PARSE YOLOv8 OUTPUT
    // ====================================================================

    void ParseYoloV8Output(Tensor<float> output)
    {
        currentDetections.Clear();

        var shape = output.shape;
        int rank = shape.rank;
        int total = shape.length;

        int numCoords = 4;
        int expectedChannels = numCoords + numClasses;

        int channels = 0;
        int anchors = 0;
        bool channelsFirst = false;

        // Determine layout
        if (rank == 3)
        {
            int d0 = shape[0];
            int d1 = shape[1];
            int d2 = shape[2];

            if (d0 != 1)
            {
                Debug.LogWarning($"RunYOLO_AR: Unexpected batch dim {d0}");
                return;
            }

            if (d1 == expectedChannels)
            {
                channels = d1;
                anchors = d2;
                channelsFirst = true;
            }
            else if (d2 == expectedChannels)
            {
                channels = d2;
                anchors = d1;
                channelsFirst = false;
            }
            else
            {
                Debug.LogWarning($"RunYOLO_AR: Unexpected 3D output shape [{d0}, {d1}, {d2}]");
                return;
            }
        }
        else if (rank == 1)
        {
            if (total % expectedChannels != 0)
            {
                Debug.LogWarning($"RunYOLO_AR: Flat output length {total} not divisible by {expectedChannels}");
                return;
            }

            channels = expectedChannels;
            anchors = total / expectedChannels;
            channelsFirst = false;
        }
        else
        {
            Debug.LogWarning($"RunYOLO_AR: Unexpected output rank {rank}");
            return;
        }

        LogDebug($"RunYOLO_AR: Parsing (rank={rank}, channels={channels}, anchors={anchors})");

        // Helper function to read values
        float GetValue(int anchorIndex, int channelIndex)
        {
            if (rank == 3)
            {
                return channelsFirst 
                    ? output[0, channelIndex, anchorIndex]
                    : output[0, anchorIndex, channelIndex];
            }
            else
            {
                int flatIndex = anchorIndex * channels + channelIndex;
                return output[flatIndex];
            }
        }

        // Decode detections
        for (int a = 0; a < anchors; a++)
        {
            float cx = GetValue(a, 0);
            float cy = GetValue(a, 1);
            float w = GetValue(a, 2);
            float h = GetValue(a, 3);

            int bestClass = -1;
            float bestScore = 0f;

            for (int c = 0; c < numClasses; c++)
            {
                float clsScore = GetValue(a, numCoords + c);
                if (clsScore > bestScore)
                {
                    bestScore = clsScore;
                    bestClass = c;
                }
            }

            float confidence = bestScore;

            if (confidence < confidenceThreshold || bestClass < 0)
                continue;

            float x = cx - w * 0.5f;
            float y = cy - h * 0.5f;

            var box = new BoundingBox
            {
                x = x,
                y = y,
                width = w,
                height = h,
                confidence = confidence,
                classIndex = bestClass,
                label = (labels != null && bestClass < labels.Length) 
                    ? labels[bestClass] 
                    : $"cls_{bestClass}"
            };

            currentDetections.Add(box);
        }

        LogDebug($"RunYOLO_AR: Parsed {currentDetections.Count} raw detections.");

        // Apply Non-Maximum Suppression
        if (currentDetections.Count > 0)
        {
            var nmsDetections = ApplyNMS(currentDetections, nmsThreshold);
            currentDetections.Clear();
            currentDetections.AddRange(nmsDetections);
            Debug.Log($"RunYOLO_AR: After NMS: {currentDetections.Count} detections.");
        }

        // Project to overlay and draw
        ProjectDetectionsToOverlay();
    }

    // ====================================================================
    // 8. NON-MAXIMUM SUPPRESSION (NMS)
    // ====================================================================

    private List<BoundingBox> ApplyNMS(List<BoundingBox> boxes, float iouThreshold)
    {
        // Sort by confidence descending
        boxes.Sort((a, b) => b.confidence.CompareTo(a.confidence));

        List<BoundingBox> result = new List<BoundingBox>();
        bool[] suppressed = new bool[boxes.Count];

        for (int i = 0; i < boxes.Count; i++)
        {
            if (suppressed[i])
                continue;

            result.Add(boxes[i]);

            // Suppress overlapping boxes of the same class
            for (int j = i + 1; j < boxes.Count; j++)
            {
                if (suppressed[j])
                    continue;

                if (boxes[i].classIndex == boxes[j].classIndex)
                {
                    float iou = CalculateIoU(boxes[i], boxes[j]);
                    if (iou > iouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return result;
    }

    private float CalculateIoU(BoundingBox a, BoundingBox b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.X2, b.X2);
        float y2 = Mathf.Min(a.Y2, b.Y2);

        float intersection = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    // ====================================================================
    // 9. PROJECT TO OVERLAY & DRAW
    // ====================================================================

    void ProjectDetectionsToOverlay()
    {
        HideAllBoxes();

        if (overlayRoot == null)
        {
            Debug.LogWarning("RunYOLO_AR: overlayRoot is not assigned.");
            return;
        }

        if (currentDetections.Count == 0)
        {
            LogDebug("RunYOLO_AR: No detections to display.");
            return;
        }

        Rect rect = overlayRoot.rect;
        float canvasWidth = rect.width;
        float canvasHeight = rect.height;

        if (canvasWidth <= 0f || canvasHeight <= 0f)
        {
            Debug.LogWarning($"RunYOLO_AR: overlayRoot has invalid size: {canvasWidth}x{canvasHeight}");
            return;
        }

        Debug.Log($"RunYOLO_AR: Drawing {currentDetections.Count} boxes on canvas {canvasWidth}x{canvasHeight}");

        // Calculate scaling from model space (640x640) to canvas space
        float scaleX = canvasWidth / modelInputWidth;
        float scaleY = canvasHeight / modelInputHeight;
        float scale = Mathf.Min(scaleX, scaleY);

        float scaledWidth = modelInputWidth * scale;
        float scaledHeight = modelInputHeight * scale;
        float offsetX = (canvasWidth - scaledWidth) * 0.5f;
        float offsetY = (canvasHeight - scaledHeight) * 0.5f;

        Debug.Log($"RunYOLO_AR: Scale={scale}, Offset=({offsetX},{offsetY})");

        int boxesDrawn = 0;
        for (int i = 0; i < currentDetections.Count && i < boxUIPool.Count; i++)
        {
            var detection = currentDetections[i];
            
            // Convert from model space (0-640) to canvas space
            float x = detection.x * scale + offsetX;
            float y = detection.y * scale + offsetY;
            float w = detection.width * scale;
            float h = detection.height * scale;

            // Skip if box is too small or outside canvas
            if (w < 5f || h < 5f)
            {
                LogDebug($"RunYOLO_AR: Skipping box {i} - too small ({w}x{h})");
                continue;
            }

            if (x < 0 || y < 0 || x + w > canvasWidth || y + h > canvasHeight)
            {
                LogDebug($"RunYOLO_AR: Box {i} partially outside canvas");
            }

            UpdateBoundingBoxUI(boxUIPool[i], x, y, w, h, detection.label, detection.confidence);
            boxesDrawn++;
        }

        Debug.Log($"RunYOLO_AR: Drew {boxesDrawn} bounding boxes.");
    }

    // ====================================================================
    // 10. UI BOX CREATION AND MANAGEMENT
    // ====================================================================

    private BoundingBoxUI CreateBoundingBoxUI()
    {
        var boxUI = new BoundingBoxUI();

        // Root container
        boxUI.rootObject = new GameObject("BoundingBox");
        boxUI.rootObject.transform.SetParent(overlayRoot, false);
        
        boxUI.rectTransform = boxUI.rootObject.AddComponent<RectTransform>();
        boxUI.rectTransform.anchorMin = new Vector2(0, 1); // Top-left anchor
        boxUI.rectTransform.anchorMax = new Vector2(0, 1);
        boxUI.rectTransform.pivot = new Vector2(0, 1);

        // Create 4 lines for border (top, right, bottom, left)
        boxUI.lines = new RawImage[4];
        string[] lineNames = { "Top", "Right", "Bottom", "Left" };
        
        for (int i = 0; i < 4; i++)
        {
            GameObject lineObj = new GameObject(lineNames[i]);
            lineObj.transform.SetParent(boxUI.rootObject.transform, false);
            
            RectTransform lineRT = lineObj.AddComponent<RectTransform>();
            RawImage lineImage = lineObj.AddComponent<RawImage>();
            lineImage.color = boxColor;
            
            boxUI.lines[i] = lineImage;
        }

        // Label
        GameObject labelObj = new GameObject("Label");
        labelObj.transform.SetParent(boxUI.rootObject.transform, false);
        
        RectTransform labelRT = labelObj.AddComponent<RectTransform>();
        labelRT.anchorMin = new Vector2(0, 1);
        labelRT.anchorMax = new Vector2(0, 1);
        labelRT.pivot = new Vector2(0, 1);
        labelRT.anchoredPosition = new Vector2(5, 0);
        labelRT.sizeDelta = new Vector2(200, 30);
        
        boxUI.label = labelObj.AddComponent<Text>();
        boxUI.label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        boxUI.label.fontSize = labelFontSize;
        boxUI.label.color = boxColor;
        boxUI.label.alignment = TextAnchor.UpperLeft;
        
        // Add shadow for readability
        Shadow shadow = labelObj.AddComponent<Shadow>();
        shadow.effectColor = new Color(0, 0, 0, 0.8f);
        shadow.effectDistance = new Vector2(1, -1);

        return boxUI;
    }

    private void UpdateBoundingBoxUI(BoundingBoxUI boxUI, float x, float y, float width, float height, string label, float confidence)
    {
        boxUI.rootObject.SetActive(true);

        // Position and size the container
        boxUI.rectTransform.anchoredPosition = new Vector2(x, -y);
        boxUI.rectTransform.sizeDelta = new Vector2(width, height);

        // Update the 4 border lines
        // Top line
        RectTransform topRT = boxUI.lines[0].GetComponent<RectTransform>();
        topRT.anchorMin = new Vector2(0, 1);
        topRT.anchorMax = new Vector2(1, 1);
        topRT.pivot = new Vector2(0, 1);
        topRT.anchoredPosition = Vector2.zero;
        topRT.sizeDelta = new Vector2(0, boxLineWidth);

        // Right line
        RectTransform rightRT = boxUI.lines[1].GetComponent<RectTransform>();
        rightRT.anchorMin = new Vector2(1, 0);
        rightRT.anchorMax = new Vector2(1, 1);
        rightRT.pivot = new Vector2(1, 1);
        rightRT.anchoredPosition = Vector2.zero;
        rightRT.sizeDelta = new Vector2(boxLineWidth, 0);

        // Bottom line
        RectTransform bottomRT = boxUI.lines[2].GetComponent<RectTransform>();
        bottomRT.anchorMin = new Vector2(0, 0);
        bottomRT.anchorMax = new Vector2(1, 0);
        bottomRT.pivot = new Vector2(0, 0);
        bottomRT.anchoredPosition = Vector2.zero;
        bottomRT.sizeDelta = new Vector2(0, boxLineWidth);

        // Left line
        RectTransform leftRT = boxUI.lines[3].GetComponent<RectTransform>();
        leftRT.anchorMin = new Vector2(0, 0);
        leftRT.anchorMax = new Vector2(0, 1);
        leftRT.pivot = new Vector2(0, 1);
        leftRT.anchoredPosition = Vector2.zero;
        leftRT.sizeDelta = new Vector2(boxLineWidth, 0);

        // Update label
        boxUI.label.text = $"{label} {confidence:P0}";
    }

    private void HideAllBoxes()
    {
        foreach (var boxUI in boxUIPool)
        {
            boxUI.rootObject.SetActive(false);
        }
    }

    // ====================================================================
    // 11. UTILITY
    // ====================================================================

    [System.Diagnostics.Conditional("UNITY_EDITOR")]
    private void LogDebug(string message)
    {
        Debug.Log(message);
    }
}