using System;
using System.Linq;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using Unity.InferenceEngine;

[Serializable]
public struct YoloDetection
{
    public Rect rect;    // pixel coords in source texture space (origin bottom-left)
    public string label;
    public float score;
}

public class YoloDetectorV8 : MonoBehaviour
{
    [Header("Model & Labels")]
    public ModelAsset modelAsset;
    public TextAsset labelsFile;

    [Header("Inference")]
    public BackendType backend = BackendType.GPUCompute;
    [Tooltip("YOLOv8 input size (square). Common: 640")]
    public int inputSize = 640;

    [Header("Thresholds")]
    [Range(0,1f)] public float scoreThreshold = 0.25f;
    [Range(0,1f)] public float iouThreshold   = 0.45f;
    [Tooltip("Max boxes kept after NMS")]
    public int maxDetections = 300;

    // Runtime
    private Worker _worker;
    private Tensor<float> _inputTensor;
    private float[] _inputBuffer;
    private string[] _labels;
    private int _numClasses = -1;

    void Awake()
    {
        if (modelAsset == null)
        {
            Debug.LogError("[YoloDetectorV8] ModelAsset not assigned.");
            enabled = false; return;
        }
        if (labelsFile == null)
        {
            Debug.LogError("[YoloDetectorV8] labelsFile not assigned.");
            enabled = false; return;
        }

        _labels = Regex.Split(labelsFile.text, "\n|\r|\r\n")
                       .Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();

        var runtimeModel = ModelLoader.Load(modelAsset);
        _worker = new Worker(runtimeModel, backend);

        _inputBuffer = new float[1 * 3 * inputSize * inputSize];
        _inputTensor = new Tensor<float>(new TensorShape(1, 3, inputSize, inputSize), _inputBuffer);
    }

    void OnDestroy()
    {
        _inputTensor?.Dispose();
        _worker?.Dispose();
    }

    /// <summary>
    /// Run YOLOv8 on the given Texture2D. Returns detections in source pixel coords.
    /// </summary>
    public List<YoloDetection> Run(Texture2D src)
    {
        if (src == null) return new List<YoloDetection>();

        // 1) Preprocess (letterbox into CHW)
        var lb = PreprocessLetterboxToCHW(src, _inputBuffer, inputSize, inputSize);

        // 2) Inference
        _worker.Schedule(_inputTensor);
        var output = _worker.PeekOutput();

        // 3) Try to get output as Tensor<float>
        Tensor<float> y = output as Tensor<float>;
        if (y == null)
        {
            // Try alternative output names
            try { y = _worker.PeekOutput("output0") as Tensor<float>; }
            catch 
            { 
                try { y = _worker.PeekOutput("output") as Tensor<float>; }
                catch { Debug.LogError("[YoloDetectorV8] Could not find output tensor."); }
            }
        }

        if (y == null)
        {
            Debug.LogError("[YoloDetectorV8] Could not retrieve output tensor as Tensor<float>.");
            return new List<YoloDetection>();
        }

        var dets = ParseYoloV8(y, lb, src.width, src.height);

        // 4) NMS
        return NMS(dets, iouThreshold, maxDetections);
    }

    // ---------- Parsing YOLOv8 ----------

    private List<YoloDetection> ParseYoloV8(Tensor<float> y, LetterboxInfo lb, int srcW, int srcH)
    {
        var shp = y.shape;
        
        // Access TensorShape dimensions using indexer
        int dim0 = shp[0];  // batch
        int dim1 = shp[1];  // could be channels or height
        int dim2 = shp[2];  // could be height or width
        int dim3 = shp.rank > 3 ? shp[3] : 1;  // width or 1

        // Typical YOLOv8 detection tensor (after export):
        // Case A: [1, 84, 8400] or [1, 84, 8400, 1]  => dim1=84, dim2=8400
        // Case B: [1, 8400, 84] or [1, 8400, 84, 1]  => dim1=8400, dim2=84
        
        int big = Mathf.Max(dim1, dim2);
        int small = Mathf.Min(dim1, dim2);
        int Ncand = big;
        int D = small;  // 4 + numClasses

        if (_numClasses < 0) _numClasses = Mathf.Max(1, D - 4);

        // Download tensor to CPU - use DownloadToNativeArray or direct indexing
        bool channelsFirst = (dim1 == D && dim2 == Ncand);

        var outList = new List<YoloDetection>(Mathf.Min(Ncand, 1024));

        for (int i = 0; i < Ncand; i++)
        {
            float cx, cy, ww, hh;
            
            if (channelsFirst)
            {
                // Layout: [1, D, Ncand] or [1, D, Ncand, 1]
                cx = y[0, 0, i, 0];
                cy = y[0, 1, i, 0];
                ww = y[0, 2, i, 0];
                hh = y[0, 3, i, 0];
            }
            else
            {
                // Layout: [1, Ncand, D] or [1, Ncand, D, 1]
                cx = y[0, i, 0, 0];
                cy = y[0, i, 1, 0];
                ww = y[0, i, 2, 0];
                hh = y[0, i, 3, 0];
            }

            // Find best class
            int best = 0; 
            float bestScore = 0f;
            int clsCount = _numClasses;
            
            for (int k = 0; k < clsCount; k++)
            {
                float s;
                if (channelsFirst)
                {
                    s = y[0, 4 + k, i, 0];
                }
                else
                {
                    s = y[0, i, 4 + k, 0];
                }
                
                if (s > bestScore) { bestScore = s; best = k; }
            }

            if (bestScore < scoreThreshold) continue;

            // YOLOv8 boxes as (cx, cy, w, h) in letterboxed space
            float x0 = cx - ww * 0.5f;
            float y0 = cy - hh * 0.5f;
            float x1 = cx + ww * 0.5f;
            float y1 = cy + hh * 0.5f;

            // Undo letterbox: remove padding then divide by scale
            x0 = (x0 - lb.padX) / lb.scale;
            y0 = (y0 - lb.padY) / lb.scale;
            x1 = (x1 - lb.padX) / lb.scale;
            y1 = (y1 - lb.padY) / lb.scale;

            // Clamp to source image
            x0 = Mathf.Clamp(x0, 0, srcW - 1);
            y0 = Mathf.Clamp(y0, 0, srcH - 1);
            x1 = Mathf.Clamp(x1, 0, srcW - 1);
            y1 = Mathf.Clamp(y1, 0, srcH - 1);

            var rect = new Rect(x0, y0, Mathf.Max(1, x1 - x0), Mathf.Max(1, y1 - y0));
            string label = (best >= 0 && _labels != null && best < _labels.Length) ? _labels[best] : ("cls_" + best);
            outList.Add(new YoloDetection { rect = rect, label = label, score = bestScore });
        }

        return outList;
    }

    // ---------- NMS ----------

    private static List<YoloDetection> NMS(List<YoloDetection> dets, float iouThresh, int maxKeep)
    {
        var result = new List<YoloDetection>(Mathf.Min(dets.Count, maxKeep));
        if (dets.Count == 0) return result;

        var sorted = dets.OrderByDescending(d => d.score).ToList();
        var removed = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (removed[i]) continue;
            var a = sorted[i];
            result.Add(a);
            if (result.Count >= maxKeep) break;

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (removed[j]) continue;
                var b = sorted[j];
                if (IoU(a.rect, b.rect) > iouThresh) removed[j] = true;
            }
        }
        return result;
    }

    private static float IoU(Rect a, Rect b)
    {
        float interX0 = Mathf.Max(a.xMin, b.xMin);
        float interY0 = Mathf.Max(a.yMin, b.yMin);
        float interX1 = Mathf.Min(a.xMax, b.xMax);
        float interY1 = Mathf.Min(a.yMax, b.yMax);
        float interW = Mathf.Max(0, interX1 - interX0);
        float interH = Mathf.Max(0, interY1 - interY0);
        float inter = interW * interH;
        float union = a.width * a.height + b.width * b.height - inter;
        return union <= 0 ? 0f : inter / union;
    }

    // ---------- Preprocess (letterbox → CHW) ----------

    struct LetterboxInfo { public float scale; public float padX; public float padY; }

    private LetterboxInfo PreprocessLetterboxToCHW(Texture2D src, float[] dst, int dstW, int dstH)
    {
        int sw = src.width, sh = src.height;
        var pixels = src.GetPixels32();

        float r = Mathf.Min((float)dstW / sw, (float)dstH / sh);
        int rw = Mathf.RoundToInt(sw * r);
        int rh = Mathf.RoundToInt(sh * r);
        int padX = (dstW - rw) / 2;
        int padY = (dstH - rh) / 2;

        // zero-fill (black padding)
        int plane = dstW * dstH;
        Array.Clear(dst, 0, dst.Length);

        // nearest resize into letterbox
        for (int y = 0; y < rh; y++)
        {
            int sy = Mathf.Min(sh - 1, Mathf.RoundToInt(y / r));
            for (int x = 0; x < rw; x++)
            {
                int sx = Mathf.Min(sw - 1, Mathf.RoundToInt(x / r));
                var c = pixels[sy * sw + sx];

                int dx = padX + x;
                int dy = padY + y;
                int i = dy * dstW + dx;

                dst[i]              = c.r / 255f; // R channel
                dst[plane + i]      = c.g / 255f; // G channel
                dst[2 * plane + i]  = c.b / 255f; // B channel
            }
        }

        return new LetterboxInfo { scale = r, padX = padX, padY = padY };
    }
}